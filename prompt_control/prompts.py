from __future__ import annotations
import logging
import re
import torch
import math
from functools import partial
from comfy_extras.nodes_mask import FeatherMask, MaskComposite
from nodes import ConditioningAverage

from .utils import (
    safe_float,
    get_function,
    split_by_function,
    parse_floats,
    smarter_split,
    call_node,
    split_quotable,
    FunctionSpec,
    ComfyConditioning,
)
from .adv_encode import advanced_encode_from_tokens
from .cutoff import process_cuts
from .parser import parse_cuts

from .attention_couple_ppm import set_cond_attnmask

log = logging.getLogger("comfyui-prompt-control")

AVAILABLE_STYLES = ["comfy", "perp", "A1111", "compel", "comfy++", "down_weight"]
AVAILABLE_NORMALIZATIONS = ["none", "mean", "length", "length+mean"]

SHUFFLE_GEN = torch.Generator(device="cpu")


def get_sdxl(text, defaults):
    # Defaults fail to parse and get looked up from the defaults dict
    text, sdxl = get_function(text, "SDXL", ["none", "none", "none"])
    if not sdxl:
        return text, {}
    args = sdxl[0].args
    d = defaults
    w, h = parse_floats(args[0], [d.get("sdxl_width", 1024), d.get("sdxl_height", 1024)], split_re="\\s+")
    tw, th = parse_floats(args[1], [d.get("sdxl_twidth", 1024), d.get("sdxl_theight", 1024)], split_re="\\s+")
    cropw, croph = parse_floats(args[2], [d.get("sdxl_cwidth", 0), d.get("sdxl_cheight", 0)], split_re="\\s+")

    opts = {
        "width": int(w),
        "height": int(h),
        "target_width": int(tw),
        "target_height": int(th),
        "crop_w": int(cropw),
        "crop_h": int(croph),
    }
    return text, opts


def get_clipweights(text, existing_spec=None):
    text, spec = get_function(text, "TE_WEIGHT", defaults=None)
    if not spec:
        return existing_spec or {}, text
    args = spec[0].args[0].strip()
    res = {}
    for arg in args.split(","):
        try:
            te, val = arg.strip().split("=")
            te, val = te.strip(), float(val.strip())
            res[te] = val
        except ValueError:
            log.warning("Invalid TE weight spec '%s', ignoring...", arg.strip())
    return res, text


def get_style(text, default_style="comfy", default_normalization="none"):
    text, styles = get_function(text, "STYLE", [default_style, default_normalization])
    if not styles:
        return default_style, default_normalization, text
    style, normalization = styles[0].args
    style = style.strip()
    normalization = normalization.strip()
    if style.replace("old+", "") not in AVAILABLE_STYLES:
        log.warning("Unrecognized prompt style: %s. Using %s", style, default_style)

    for part in normalization.split("+"):
        if part not in AVAILABLE_NORMALIZATIONS:
            log.warning("Unrecognized prompt normalization: %s. Using %s", normalization, default_normalization)
            normalization = default_normalization
            break

    return style, normalization, text


def shuffle_chunk(func_spec: FunctionSpec, c: str) -> str:
    func = func_spec.name
    shuffle = func_spec.args
    shuffle_count = int(safe_float(shuffle[0], 0))
    _, separator, joiner = shuffle
    if separator == "default":
        separator = ","

    if not separator:
        separator = ","

    joiner = {
        "default": ",",
        "separator": separator,
    }.get(joiner, joiner)

    log.debug("%s arg=%s sep=%s join=%s", func, shuffle_count, separator, joiner)
    separated = smarter_split(separator, c)
    log.debug("Prompt split into %s", separated)
    if func == "SHIFT":
        shuffle_count = shuffle_count % len(separated)
        permutation = separated[shuffle_count:] + separated[:shuffle_count]
    elif func == "SHUFFLE":
        SHUFFLE_GEN.manual_seed(shuffle_count)
        permutation = [separated[i] for i in torch.randperm(len(separated), generator=SHUFFLE_GEN)]
    else:
        # ??? should never get here
        permutation = separated

    permutation = [p for p in permutation if p.strip()]
    if permutation != separated:
        c = joiner.join(permutation)
    return c


def fix_word_ids(tokens):
    """Fix word indexes. Tokenizing separately (when BREAKs exist) causes the indexes to restart which causes problems with some weighting algorithms that rely on them"""
    for key in tokens:
        max_idx = 0
        for group in range(len(tokens[key])):
            for i, token in enumerate(tokens[key][group]):
                if len(token) < 3:
                    # No need to fix ids when they don't exist
                    return tokens
                # Ignore zeros, they represent the padding token
                if token[2] != 0 and token[2] < max_idx:
                    tokens[key][group][i] = (token[0], token[1], token[2] + max_idx)
            max_idx = max(max_idx, max(x for _, _, x in tokens[key][group]))
    return tokens


def tokenize_chunks(clip, text, need_word_ids, can_break):
    chunks = list(split_quotable(text, r"\bBREAK\b"))
    token_chunks = []
    shuffled_chunks = []
    for c in chunks:
        c, shuffles = get_function(c.strip(), "(SHIFT|SHUFFLE)", ["0", "default", "default"])
        r = c
        for s in shuffles:
            r = shuffle_chunk(s, r)
        if r != c:
            log.info("Shuffled prompt chunk to %s", r)
        shuffled_chunks.append(r)
        t = clip.tokenize(c, return_word_ids=need_word_ids)
        token_chunks.append(t)

    tokens = token_chunks[0]
    full_prompt = "".join(shuffled_chunks)
    full_tokenized = tokens
    if len(chunks) > 1:
        full_tokenized = clip.tokenize(full_prompt, return_word_ids=need_word_ids)
        for key in tokens:
            if not can_break.get(key):
                log.warning("BREAK does not make sense for %s, tokenizing as one chunk. Use CAT instead.", key)
                tokens[key] = full_tokenized[key]
                continue
            for c in token_chunks[1:]:
                tokens[key].extend(c[key])

    return tokens


def tokenize(clip, text, can_break, empty_tokens):
    # defaults=None means there is no argument parsing at all
    text, l_prompts = get_function(text, "CLIP_L", defaults=None)
    text, te_prompts = get_function(text, "TE", defaults=None)
    need_word_ids = True
    tokens = tokenize_chunks(clip, text, need_word_ids, can_break)

    per_te_prompts = {}
    if l_prompts:
        log.warning("Note: CLIP_L is deprecated. Use TE(l=prompt) instead")
        per_te_prompts["l"] = [x.args for x in l_prompts]

    for prompt in te_prompts:
        prompt = prompt.args[0]
        if prompt.strip() == "help":
            log.info("Encoders available for TE: %s", ", ".join(tokens.keys()))
            continue
        params = prompt.split("=", 1)
        if len(params) != 2:
            log.warning("Invalid TE call, ignoring: %s", prompt)
            continue
        te = params[0].strip()
        prompt = params[1].strip()
        if te not in tokens:
            log.warning("Invalid TE call, no TE with key '%s', ignoring: %s", te)
            log.info("Encoders available for TE: %s", ", ".join(tokens.keys()))
            continue
        l = per_te_prompts.get(te, [])
        l.append(prompt)
        per_te_prompts[te] = l

    if per_te_prompts:
        for key in per_te_prompts:
            prompt = " ".join(per_te_prompts[key])
            tokens[key] = tokenize_chunks(clip, prompt, need_word_ids, can_break)[key]
            log.info("Encoded prompt with TE '%s': %s", key, prompt)

    maxlen = max([0] + [len(tokens[k]) for k in tokens if can_break[k]])
    for k in tokens:
        if not can_break[k]:
            continue
        while len(tokens[k]) < maxlen:
            tokens[k] += empty_tokens[k]

    return fix_word_ids(tokens)


def encode_prompt_segment(
    clip,
    text,
    settings,
    default_style="comfy",
    default_normalization="none",
    clip_weights=None,
) -> list[ComfyConditioning]:
    style, normalization, text = get_style(text, default_style, default_normalization)
    clip_weights, text = get_clipweights(text, clip_weights)
    text, cuts = parse_cuts(text)
    extra = {}
    if clip_weights:
        extra["clip_weights"] = clip_weights
    if cuts:
        extra["cuts"] = cuts

    empty = clip.tokenize("", return_word_ids=True)
    can_break = {}
    for k in empty:
        tokenizer = getattr(clip.tokenizer, f"clip_{k}", getattr(clip.tokenizer, k, None))
        can_break[k] = tokenizer and tokenizer.pad_to_max_length

    clip = hook_te(clip, empty.keys(), style, normalization, extra)

    # Chunks to ConditioningAverage:

    text, averages = split_by_function(text, "AVG", ["0.5"], require_args=False)
    prompts_to_avg = []
    for chunk, avg in averages:
        w = safe_float(avg.args[0], 0.5)
        prompts_to_avg.append((text, w))
        text = chunk
    prompts_to_avg.append((text, 1.0))

    conds_to_avg = []
    for prompt, weight in prompts_to_avg:
        conds_to_cat = []
        for c in split_quotable(prompt, r"\bCAT\b"):
            tokens = tokenize(clip, c, can_break, empty)
            conds_to_cat.append(clip.encode_from_tokens_scheduled(tokens, add_dict=settings))

        base = conds_to_cat[0]
        for cond in conds_to_cat[1:]:
            assert len(cond) == len(base), "Conditioning length mismatch"
            # Pooled gets ignored
            for i in range(len(base)):
                c1 = base[i][0]
                c2 = cond[i][0]
                base[i][0] = torch.cat((c1, c2), 1)
        conds_to_avg.append((base, weight))

    base, w = conds_to_avg[0]
    for cond, next_w in conds_to_avg[1:]:
        assert len(base) == len(cond), "Conditioning length mismatch"
        if w == 1.0:
            w = next_w
            continue
        for i in range(len(base)):
            (cond,) = call_node(ConditioningAverage, [base[i]], [cond[i]], w)
            base[i] = cond[0]
        w = next_w

    return base


def calc_w(tensor, w):
    if math.isclose(w, 0):
        return torch.zeros_like(tensor)
    elif math.isclose(w, 1.0):
        return tensor
    else:
        return tensor * w


def apply_weights(output, te_name, spec):
    """Applies weights to TE outputs"""
    if not spec:
        return output

    if te_name.startswith("clip_"):
        te_name = te_name[5:]

    default = spec.get("all", None)

    if isinstance(output, tuple):
        out, pooled, *extra = output
        pkey = te_name + "_pooled"
        if te_name in spec or pkey in spec or default is not None:
            w = spec.get(te_name, default)
            pooled_w = spec.get(pkey, w)
            if w is None:
                w = 1.0
            if pooled_w is None:
                pooled_w = 1.0
            log.info("Weighting %s output by %s, pooled by %s", te_name, w, pooled_w)
            out = calc_w(out, w)
            if pooled is not None:
                pooled = calc_w(pooled, pooled_w)

        return (out, pooled) + tuple(extra)
    else:
        if te_name in spec or default is not None:
            w = spec.get(te_name, default)
            log.info("Weighting %s output by %s", te_name, w)
            output = calc_w(output, w)
        return output


def make_patch(te_name, orig_fn, normalization, style, extra):
    def encode(t):
        r = advanced_encode_from_tokens(
            t, normalization, style, orig_fn, return_pooled=True, apply_to_pooled=False, **extra
        )
        return apply_weights(r, te_name, extra.get("clip_weights"))

    if "cuts" in extra:
        return partial(process_cuts, encode, extra)
    return encode


def hook_te(clip, te_names, style, normalization, extra):
    if style == "comfy" and normalization == "none" and not extra:
        return clip
    newclip = clip.clone()
    for te_name in te_names:
        tokenizer = getattr(clip.tokenizer, f"clip_{te_name}", getattr(clip.tokenizer, te_name, None))
        if tokenizer:
            x = extra.copy()
            x["tokenizer"] = tokenizer
            if not hasattr(clip.patcher.model, te_name):
                te_name = "clip_" + te_name
            if not hasattr(clip.patcher.model, te_name):
                log.warning("TE model %s not found on model patcher. Skipping...", te_name)
                continue

            log.debug("Hooked into te=%s with style=%s, normalization=%s", te_name, style, normalization)
            encode = clip.patcher.get_model_object(f"{te_name}.encode_token_weights")
            x["has_negpip"] = clip.patcher.model_options.get("ppm_negpip", False)
            newclip.patcher.add_object_patch(
                f"{te_name}.encode_token_weights",
                make_patch(
                    te_name,
                    encode,
                    normalization,
                    style,
                    x,
                ),
            )
        # 'g' and 'l' exist in these are clip_g and clip_l
        else:
            log.warning("Tokens contain items with key %s but no tokenizer found on object with that name.", te_name)
    return newclip


def get_area(text):
    text, areas = get_function(text, "AREA", ["0 1", "0 1", "1"])
    if not areas:
        return text, None

    args = areas[0].args
    x, w = parse_floats(args[0], [0.0, 1.0], split_re="\\s+")
    y, h = parse_floats(args[1], [0.0, 1.0], split_re="\\s+")
    weight = safe_float(args[2], 1.0)

    def is_pct(f):
        return f >= 0.0 and f <= 1.0

    def is_pixel(f):
        return f == 0 or f > 1

    if all(is_pct(v) for v in [h, w, y, x]):
        area = ("percentage", h, w, y, x)
    elif all(is_pixel(v) for v in [h, w, y, x]):
        area = (int(h) // 8, int(w) // 8, int(y) // 8, int(x) // 8)
    else:
        raise Exception(
            f"AREA specified with invalid size {x} {w}, {h} {y}. They must either all be percentages between 0 and 1 or positive integer pixel values excluding 1"
        )

    return text, (area, weight)


def get_mask_size(text, defaults):
    text, sizes = get_function(text, "MASK_SIZE", ["512", "512"])
    if not sizes:
        return text, (defaults.get("mask_width", 512), defaults.get("mask_height", 512))
    w, h = sizes[0].args
    return text, (int(w), int(h))


def make_mask(args, size, weight):
    x1, x2 = parse_floats(args[0], [0.0, 1.0], split_re="\\s+")
    y1, y2 = parse_floats(args[1], [0.0, 1.0], split_re="\\s+")

    def is_pct(f):
        return f >= 0.0 and f <= 1.0

    def is_pixel(f):
        return f == 0 or f > 1

    if all(is_pct(v) for v in [x1, x2, y1, y2]):
        w, h = size
        xs = int(w * x1), int(w * x2)
        ys = int(h * y1), int(h * y2)
    elif all(is_pixel(v) for v in [x1, x2, y1, y2]):
        w, h = size
        xs = int(x1), int(x2)
        ys = int(y1), int(y2)
    else:
        raise Exception(
            f"MASK specified with invalid size {x1} {x2}, {y1} {y2}. They must either all be percentages between 0 and 1 or positive integer pixel values excluding 1"
        )

    mask = torch.full((h, w), 0, dtype=torch.float32, device="cpu")
    mask[ys[0] : ys[1], xs[0] : xs[1]] = weight
    mask = mask.unsqueeze(0)
    log.debug("Mask xs=%s, ys=%s, shape=%s, weight=%s", xs, ys, mask.shape, weight)
    return mask


def get_mask(text, size, input_masks):
    """Parse MASK(x1 x2, y1 y2, weight), IMASK(i, weight) and FEATHER(left top right bottom)"""
    # TODO: combine multiple masks
    text, masks = get_function(text, "MASK", ["0 1", "0 1", "1", "multiply"])
    text, imasks = get_function(text, "IMASK", ["0", "1", "multiply"])
    text, feathers = get_function(text, "FEATHER", ["0 0 0 0"])
    text, maskw = get_function(text, "MASKW", ["1.0"])
    if not masks and not imasks:
        return text, None, None

    def feather(f, mask):
        l, t, r, b, *_ = [int(x) for x in parse_floats(f[0], [0, 0, 0, 0], split_re="\\s+")]
        mask = call_node(FeatherMask, mask, l, t, r, b)[0]
        log.info("FeatherMask l=%s, t=%s, r=%s, b=%s", l, t, r, b)
        return mask

    mask = None
    totalweight = 1.0
    if maskw:
        totalweight = safe_float(maskw[0].args[0], 1.0)
    i = 0
    for m in masks:
        weight = safe_float(m.args[2], 1.0)
        op = m.args[3]
        nextmask = make_mask(m.args, size, weight)
        if i < len(feathers):
            nextmask = feather(feathers[i].args, nextmask)
        i += 1
        if mask is not None:
            log.info("MaskComposite op=%s", op)
            mask = call_node(MaskComposite, mask, nextmask, 0, 0, op)[0]
        else:
            mask = nextmask

    for im in imasks:
        idx, w, op = im.args
        idx = int(safe_float(idx, 0.0))
        w = safe_float(w, 1.0)
        if input_masks is None:
            log.warn(
                "IMASK requires you to attach custom masks to the CLIP object using PCAddMasksToClIP before using it"
            )
            input_masks = []

        if len(input_masks) < idx + 1:
            log.warn("IMASK index %s not found, ignoring...", idx)
            continue
        nextmask = input_masks[idx] * w
        if i < len(feathers):
            nextmask = feather(feathers[i].args, nextmask)
        i += 1
        if mask is not None:
            mask = call_node(MaskComposite, mask, nextmask, 0, 0, op)[0]
        else:
            mask = nextmask

    # apply leftover FEATHER() specs to the whole
    for f in feathers[i:]:
        mask = feather(f.args, mask)

    return text, mask, totalweight


def get_noise(text):
    text, noises = get_function(
        text,
        "NOISE",
        ["0.0", "none"],
    )
    if not noises:
        return text, None, None
    w = 0
    # Only take seed from first noise spec, for simplicity
    seed = noises[0].args[0].strip()
    if seed == "none":
        gen = None
    else:
        seed = safe_float(seed, 0)
        gen = torch.Generator()
        gen.manual_seed(int(seed))
    for n in noises:
        w += safe_float(n.args[0], 0.0)
    return text, max(min(w, 1.0), 0.0), gen


def apply_noise(cond, weight, gen):
    if cond is None or not weight:
        return cond

    n = torch.randn(cond.size(), generator=gen).to(cond)

    return cond * (1 - weight) + n * weight


def process_settings(prompt, defaults, masks, mask_size, sdxl_opts):
    if "ATTN()" in prompt:
        raise ValueError("ATTN() no longer works and has been replaced by COUPLE()")

    def weight(t):
        opts = {}
        m = re.search(r":(-?\d\.?\d*)(![A-Za-z]+)?$", t.strip())
        if not m:
            return (None, opts, t)
        w = float(m[1])
        tag = m[2]
        t = t[: m.span()[0]]
        if tag == "!noscale":
            opts["scale"] = 1

        return w, opts, t

    settings = {"prompt": prompt}

    if "FILL()" in prompt:
        prompt = prompt.replace("FILL()", "")
        settings["x-promptcontrol.fill"] = True
    prompt, mask, mask_weight = get_mask(prompt, mask_size, masks)
    prompt, noise_w, generator = get_noise(prompt)
    prompt, area = get_area(prompt)
    prompt, local_sdxl_opts = get_sdxl(prompt, defaults)
    # Get weight last so other syntax doesn't interfere with it
    w, opts, prompt = weight(prompt)
    if w is not None:
        settings["strength"] = w
    settings.update(sdxl_opts)
    settings.update(local_sdxl_opts)
    if area:
        settings["area"] = area[0]
        settings["strength"] = area[1]
        settings["set_area_to_bounds"] = False
    if mask is not None:
        settings["mask"] = mask
        settings["mask_strength"] = mask_weight

    return prompt, settings


def encode_prompt(clip, text, start_pct, end_pct, defaults, masks):
    # First style modifier applies to ANDed prompts too unless overridden
    style, normalization, text = get_style(text)
    text, mask_size = get_mask_size(text, defaults)

    prompts = list(split_quotable(text, r"\bAND\b"))

    p, sdxl_opts = get_sdxl(prompts[0], defaults)
    prompts[0] = p

    conds = []
    # TODO: is this still needed?
    # scale = sum(abs(weight(p)[0]) for p in prompts if not ("AREA(" in p or "MASK(" in p))

    def ensure_mask(c):
        if "mask" not in c[1]:
            _, mask, _ = get_mask("MASK()", mask_size, masks)
            c[1]["mask"] = mask
            c[1]["mask_strength"] = 1.0
        return c

    def couple_mask(args):
        if args is None:
            return ""
        return f"MASK({args})"

    for prompt in prompts:
        base_prompt, attn_couple_prompts = split_by_function(prompt, "COUPLE", defaults=None, require_args=False)

        prompts = [base_prompt] + [couple_mask(f.args) + chunk for (chunk, f) in attn_couple_prompts]
        encoded = []
        for p in prompts:
            p, settings = process_settings(p, defaults, masks, mask_size, sdxl_opts)
            if settings.get("strength") == 0:  # weight is explicitly set to 0, skip
                continue
            settings["start_percent"] = start_pct
            settings["end_percent"] = end_pct
            x = encode_prompt_segment(clip, p, settings, style, normalization)
            encoded.append(x)

        assert all(
            len(c) == len(encoded[0]) for c in encoded
        ), "All encoded prompts didn't produce the same number of conds, I don't know what to do in this situation."

        # each call to encode_prompt_segment can produce a number of conds based on any
        # scheduled LoRA hooks on the clip model. Zip them together with coupled prompts
        base_cond = []
        for base_cond, *attention_couple in zip(*encoded):
            s = base_cond[1]
            # If there are LoRAs on the CLIP, we need to fix start_percent and end_percent on the new conds for things to work properly.
            s["start_percent"] = s.get("clip_start_percent", s["start_percent"])
            s["end_percent"] = s.get("clip_end_percent", s["end_percent"])
            s.pop("clip_start_percent", None)
            s.pop("clip_end_percent", None)
            base_cond = [base_cond]
            if attention_couple:
                fill = base_cond[0][1].get("x-promptcontrol.fill")
                if not fill:
                    ensure_mask(base_cond[0])
                # else, set_cond_attnmask will have the base mask fill any unspecified areas
                base_cond = set_cond_attnmask(
                    base_cond,
                    [ensure_mask(c) for c in attention_couple],
                    fill=fill,
                )
        conds.extend(base_cond)

    return conds
