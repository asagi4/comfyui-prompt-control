import logging
import re
import torch
from functools import partial
from comfy_extras.nodes_mask import FeatherMask, MaskComposite

from .utils import safe_float, get_function, parse_floats, smarter_split
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
    args = sdxl[0]
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
    args = spec[0].strip()
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
    style, normalization = styles[0]
    style = style.strip()
    normalization = normalization.strip()
    if style not in AVAILABLE_STYLES:
        log.warning("Unrecognized prompt style: %s. Using %s", style, default_style)
        style = default_style

    if normalization not in AVAILABLE_NORMALIZATIONS:
        log.warning("Unrecognized prompt normalization: %s. Using %s", normalization, default_normalization)
        normalization = default_normalization

    return style, normalization, text


def shuffle_chunk(shuffle, c):
    func, shuffle = shuffle
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


def tokenize_chunks(clip, text, need_word_ids):
    chunks = re.split(r"\bBREAK\b", text)
    token_chunks = []
    for c in chunks:
        c, shuffles = get_function(c.strip(), "(SHIFT|SHUFFLE)", ["0", "default", "default"], return_func_name=True)
        r = c
        for s in shuffles:
            r = shuffle_chunk(s, r)
        if r != c:
            log.info("Shuffled prompt chunk to %s", r)
            c = r
        t = clip.tokenize(c, return_word_ids=need_word_ids)
        token_chunks.append(t)
    tokens = token_chunks[0]

    for key in tokens:
        for c in token_chunks[1:]:
            tokens[key].extend(c[key])

    return tokens


def encode_prompt_segment(
    clip,
    text,
    settings,
    default_style="comfy",
    default_normalization="none",
    clip_weights=None,
) -> list[tuple[torch.Tensor, dict[str]]]:
    style, normalization, text = get_style(text, default_style, default_normalization)
    clip_weights, text = get_clipweights(text, clip_weights)
    text, cuts = parse_cuts(text)
    extra = {}
    if clip_weights:
        extra["clip_weights"] = clip_weights
    if cuts:
        extra["cuts"] = cuts

    # defaults=None means there is no argument parsing at all
    text, l_prompts = get_function(text, "CLIP_L", defaults=None)
    text, te_prompts = get_function(text, "TE", defaults=None)
    need_word_ids = True
    tokens = tokenize_chunks(clip, text, need_word_ids)

    per_te_prompts = {}
    if l_prompts:
        log.warning("Note: CLIP_L is deprecated. Use TE(l=prompt) instead")
        per_te_prompts["l"] = l_prompts

    for prompt in te_prompts:
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
            tokens[key] = tokenize_chunks(clip, prompt, need_word_ids)[key]
            log.info("Encoded prompt with TE '%s': %s", key, prompt)

    maxlen = max(len(tokens[k]) for k in tokens)
    empty = None
    for k in tokens:
        while len(tokens[k]) < maxlen:
            if empty is None:
                empty = clip.tokenize("", return_word_ids=need_word_ids)
            tokens[k] += empty[k]

    tokens = fix_word_ids(tokens)

    tes = []
    for k in tokens:
        if k in ["g", "l"]:
            tes.append(f"clip_{k}")
        else:
            tes.append(k)

    clip = hook_te(clip, tes, style, normalization, extra)

    return clip.encode_from_tokens_scheduled(tokens, add_dict=settings)


def apply_weights(output, te_name, spec):
    """Applies weights to TE outputs"""
    if not spec:
        return output

    if te_name.startswith("clip_"):
        te_name = te_name[5:]

    default = spec.get("all", None)

    if isinstance(output, tuple):
        out, pooled = output
        pkey = te_name + "_pooled"
        if te_name in spec or pkey in spec or default is not None:
            w = spec.get(te_name, default)
            pooled_w = spec.get(pkey, w)
            if w is None:
                w = 1.0
            if pooled_w is None:
                pooled_w = 1.0
            log.info("Weighting %s output by %s, pooled by %s", te_name, w, pooled_w)
            out = out * w
            pooled = pooled * pooled_w

        return out, pooled
    else:
        if te_name in spec or default is not None:
            w = spec.get(te_name, default)
            log.info("Weighting %s output by %s", te_name, w)
            output = output * w
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
        if hasattr(clip.patcher.model, te_name):
            x = extra.copy()
            x["tokenizer"] = getattr(clip.tokenizer, te_name)
            log.debug("Hooked into %s with style=%s, normalization=%s", te_name, style, normalization)
            newclip.patcher.add_object_patch(
                f"{te_name}.encode_token_weights",
                make_patch(
                    te_name,
                    clip.patcher.get_model_object(f"{te_name}.encode_token_weights"),
                    normalization,
                    style,
                    x,
                ),
            )
        # 'g' and 'l' exist in these are clip_g and clip_l
        else:
            log.debug("Tokens contain items with key %s but no TE found on object with that name.", te_name)
    return newclip


def get_area(text):
    text, areas = get_function(text, "AREA", ["0 1", "0 1", "1"])
    if not areas:
        return text, None

    args = areas[0]
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
    w, h = sizes[0]
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
    log.info("Mask xs=%s, ys=%s, shape=%s, weight=%s", xs, ys, mask.shape, weight)
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
        mask = FeatherMask().feather(mask, l, t, r, b)[0]
        log.info("FeatherMask l=%s, t=%s, r=%s, b=%s", l, t, r, b)
        return mask

    mask = None
    totalweight = 1.0
    if maskw:
        totalweight = safe_float(maskw[0][0], 1.0)
    i = 0
    for m in masks:
        weight = safe_float(m[2], 1.0)
        op = m[3]
        nextmask = make_mask(m, size, weight)
        if i < len(feathers):
            nextmask = feather(feathers[i], nextmask)
        i += 1
        if mask is not None:
            log.info("MaskComposite op=%s", op)
            mask = MaskComposite().combine(mask, nextmask, 0, 0, op)[0]
        else:
            mask = nextmask

    for idx, w, op in imasks:
        idx = int(safe_float(idx, 0.0))
        w = safe_float(w, 1.0)
        if len(input_masks) < idx + 1:
            log.warn("IMASK index %s not found, ignoring...", idx)
            continue
        nextmask = input_masks[idx] * w
        if i < len(feathers):
            nextmask = feather(feathers[i], nextmask)
        i += 1
        if mask is not None:
            mask = MaskComposite().combine(mask, nextmask, 0, 0, op)[0]
        else:
            mask = nextmask

    # apply leftover FEATHER() specs to the whole
    for f in feathers[i:]:
        mask = feather(f, mask)

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
    seed = safe_float(noises[0][1], "none")
    if seed == "none":
        gen = None
    else:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
    for n in noises:
        w += safe_float(n[0], 0.0)
    return text, max(min(w, 1.0), 0.0), gen


def apply_noise(cond, weight, gen):
    if cond is None or not weight:
        return cond

    n = torch.randn(cond.size(), generator=gen).to(cond)

    return cond * (1 - weight) + n * weight


def encode_prompt(clip, text, start_pct, end_pct, defaults, masks):
    # First style modifier applies to ANDed prompts too unless overridden
    style, normalization, text = get_style(text)
    text, mask_size = get_mask_size(text, defaults)

    prompts = [p.strip() for p in re.split(r"\bAND\b", text)]

    p, sdxl_opts = get_sdxl(prompts[0], defaults)
    prompts[0] = p

    def weight(t):
        opts = {}
        m = re.search(r":(-?\d\.?\d*)(![A-Za-z]+)?$", t)
        if not m:
            return (1.0, opts, t)
        w = float(m[1])
        tag = m[2]
        t = t[: m.span()[0]]
        if tag == "!noscale":
            opts["scale"] = 1

        return w, opts, t

    conds = []
    # TODO: is this still needed?
    # scale = sum(abs(weight(p)[0]) for p in prompts if not ("AREA(" in p or "MASK(" in p))
    attnmasked_prompts = []
    fill = False
    for prompt in prompts:
        attn_couple = False
        prompt_has_fill = False
        if "ATTN()" in prompt:
            prompt = prompt.replace("ATTN()", "")
            attn_couple = True
        if "FILL()" in prompt:
            prompt = prompt.replace("FILL()", "")
            prompt_has_fill = True
        prompt, mask, mask_weight = get_mask(prompt, mask_size, masks)
        text, noise_w, generator = get_noise(text)
        prompt, area = get_area(prompt)
        prompt, local_sdxl_opts = get_sdxl(prompt, defaults)
        # Get weight last so other syntax doesn't interfere with it
        w, opts, prompt = weight(prompt)
        if not w:
            continue
        settings = {"prompt": prompt}
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

        settings["start_percent"] = start_pct
        settings["end_percent"] = end_pct

        x = encode_prompt_segment(clip, prompt, settings, style, normalization)
        if attn_couple:
            if prompt_has_fill:
                if attnmasked_prompts:
                    log.warning("FILL() can only be used for the first prompt, ignoring")
                elif mask is not None:
                    log.warning("MASK() and FILL() can't be used together, ignoring FILL()")
                else:
                    fill = True
            log.info("Using attention masking for prompt segment")
            attnmasked_prompts.extend(x)
        else:
            conds.extend(x)

    def ensure_mask(c):
        if "mask" not in c[1]:
            _, mask, _ = get_mask("MASK()", mask_size, masks)
            c[1]["mask"] = mask
            c[1]["mask_strength"] = 1.0
        return c

    if attnmasked_prompts:
        base_cond = attnmasked_prompts[0]
        if not fill:
            ensure_mask(base_cond)
        # else, set_cond_attnmask will have the base mask fill any unspecified areas
        base_cond = [base_cond]
        if len(attnmasked_prompts) > 1:
            base_cond = set_cond_attnmask(
                base_cond,
                [ensure_mask(c) for c in attnmasked_prompts[1:]],
                fill=fill,
            )
        else:
            log.warning("You must specify at least two prompt segments with ATTN() for attention couple to work")
        conds.extend(base_cond)

    return conds
