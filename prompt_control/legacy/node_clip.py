import logging
import re
import torch
from ..parser import parse_prompt_schedules, parse_cuts
from .utils import Timer, equalize, apply_loras_from_spec
from ..utils import safe_float, get_function, parse_floats  # non-legacy
from .perp_weight import perp_encode
from comfy_extras.nodes_mask import FeatherMask, MaskComposite
from node_helpers import conditioning_set_values


log = logging.getLogger("comfyui-prompt-control")

try:
    from custom_nodes.ComfyUI_ADV_CLIP_emb.adv_encode import (
        advanced_encode_from_tokens,
        encode_token_weights_l,
        encode_token_weights_g,
        prepareXL,
        encode_token_weights,
    )

    have_advanced_encode = True
    AVAILABLE_STYLES = ["comfy", "A1111", "compel", "comfy++", "down_weight"]
    AVAILABLE_NORMALIZATIONS = ["none", "mean", "length", "length+mean"]
except ImportError:
    have_advanced_encode = False
    AVAILABLE_STYLES = ["comfy"]
    AVAILABLE_NORMALIZATIONS = ["none"]

try:
    from custom_nodes.Vector_Sculptor_ComfyUI.nodes import vector_sculptor_tokens

    can_sculpt = True
    log.info("Vector sculptor extension detected, can use SCULPT()")
except ImportError:
    can_sculpt = False


AVAILABLE_STYLES.append("perp")
log.info("Use STYLE(weight_interpretation, normalization) at the start of a prompt to use advanced encodings")
log.info("Weight interpretations available: %s", ",".join(AVAILABLE_STYLES))
log.info("Normalization types available: %s", ",".join(AVAILABLE_NORMALIZATIONS))


def linear_interpolate_cond(
    start, end, from_step=0.0, to_step=1.0, step=0.1, start_at=None, end_at=None, prompt_start="N/A", prompt_end="N/A"
):
    count = min(len(start), len(end))
    if len(start) != len(end):
        log.info(
            "Length of conds to interpolate does not match (start=%s != end=%s), interpolating up to %s.",
            len(start),
            len(end),
            count,
        )

    all_res = []
    for idx in range(count):
        res = []
        from_cond, to_cond = equalize(start[idx][0], end[idx][0])
        from_pooled = start[idx][1].get("pooled_output")
        to_pooled = end[idx][1].get("pooled_output")
        start_at = start_at if start_at is not None else from_step
        end_at = end_at if end_at is not None else to_step
        total_steps = int(round((to_step - from_step) / step, 0))
        num_steps = int(round((end_at - from_step) / step, 0))
        start_on = int(round((start_at - from_step) / step, 0))
        start_pct = start_at
        log.debug(
            f"interpolate_cond {idx=} {from_step=} {to_step=} {start_at=} {end_at=} {total_steps=} {num_steps=} {start_on=} {step=}"
        )
        x = 1 / (total_steps + 1)
        for s in range(start_on, num_steps):
            factor = round((s + 1) * x, 2)
            new_cond = from_cond + (to_cond - from_cond) * factor
            if from_pooled is not None and to_pooled is not None:
                from_pooled, to_pooled = equalize(from_pooled, to_pooled)
                new_pooled = from_pooled + (to_pooled - from_pooled) * factor
            elif from_pooled is not None:
                new_pooled = from_pooled

            n = [new_cond, start[idx][1].copy()]
            if new_pooled is not None:
                n[1]["pooled_output"] = new_pooled
            n[1]["start_percent"] = round(start_pct, 2)
            n[1]["end_percent"] = min(round((start_pct + step), 2), 1.0)
            start_pct += step
            start_pct = round(start_pct, 2)
            if prompt_start:
                n[1]["prompt"] = f"linear:{round(1.0 - factor, 2)} / {factor}"
            log.debug(
                "Interpolating at step %s with factor %s (%s, %s)...",
                s,
                factor,
                n[1]["start_percent"],
                n[1]["end_percent"],
            )
            res.append(n)
        if res:
            res[-1][1]["end_percent"] = round(end_at, 2)
            all_res.extend(res)
    return all_res


def get_control_points(schedule, steps, encoder):
    assert len(steps) > 1
    new_steps = set(steps)

    for step in (s[0] for s in schedule if s[0] >= steps[0] and s[0] <= steps[-1]):
        new_steps.add(step)
    control_points = [(s, encoder(schedule.at_step(s)[1])) for s in new_steps]
    log.debug("Actual control points for interpolation: %s (from %s)", new_steps, steps)
    return sorted(control_points, key=lambda x: x[0])


def linear_interpolator(control_points, step, start_pct, end_pct):
    o_start, start = control_points[0]
    o_end, _ = control_points[-1]
    t_start = o_start
    conds = []
    for t_end, end in control_points[1:]:
        if t_start < start_pct:
            t_start, start = t_end, end
            continue
        if t_start >= end_pct:
            break
        cs = linear_interpolate_cond(start, end, o_start, o_end, step, start_at=t_start, end_at=end_pct)
        if cs:
            conds.extend(cs)
        else:
            break
        t_start = t_end
        start = end
    return conds


class ScheduleToCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP",), "prompt_schedule": ("PROMPT_SCHEDULE",)},
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol/_legacy"
    FUNCTION = "apply"

    def apply(self, clip, prompt_schedule):
        with Timer("ScheduleToCond"):
            r = (control_to_clip_common(clip, prompt_schedule),)
        return r


class EditableCLIPEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {"filter_tags": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol/_donotuse"
    FUNCTION = "parse"

    def parse(self, clip, text, filter_tags=""):
        parsed = parse_prompt_schedules(text).with_filters(filter_tags)
        return (control_to_clip_common(clip, parsed),)


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


def encode_regions(clip, tokens, regions, weight_interpretation="comfy", token_normalization="none"):
    from custom_nodes.ComfyUI_Cutoff.cutoff import CLIPSetRegion, finalize_clip_regions

    clip_regions = {
        "clip": clip,
        "base_tokens": tokens,
        "regions": [],
        "targets": [],
        "weights": [],
    }

    strict_mask = 1.0
    start_from_masked = 1.0
    mask_token = ""

    for region in regions:
        region_text, target_text, w, sm, sfm, mt = region
        if w is not None:
            w = safe_float(w, 0)
        else:
            w = 1.0
        if sm is not None:
            strict_mask = safe_float(sm, 1.0)
        if sfm is not None:
            start_from_masked = safe_float(sfm, 1.0)
        if mt is not None:
            mask_token = mt
        log.info("Region: text %s, target %s, weight %s", region_text.strip(), target_text.strip(), w)
        (clip_regions,) = CLIPSetRegion.add_clip_region(None, clip_regions, region_text, target_text, w)
    log.info("Regions: mask_token=%s strict_mask=%s start_from_masked=%s", mask_token, strict_mask, start_from_masked)

    (r,) = finalize_clip_regions(
        clip_regions, mask_token, strict_mask, start_from_masked, token_normalization, weight_interpretation
    )
    cond, pooled = r[0][0], r[0][1].get("pooled_output")
    return cond, pooled


SHUFFLE_GEN = torch.Generator(device="cpu")


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

    log.info("%s arg=%s sep=%s join=%s", func, shuffle_count, separator, joiner)
    separated = c.split(separator)
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


def encode_prompt(clip, text, default_style="comfy", default_normalization="none"):
    style, normalization, text = get_style(text, default_style, default_normalization)
    sculpts = []
    if can_sculpt:
        text, sculpts = get_function(text, "SCULPT", ["1.0", "forward", "none"])
    text, regions = parse_cuts(text)
    # defaults=None means there is no argument parsing at all
    text, l_prompts = get_function(text, "CLIP_L", defaults=None)
    chunks = re.split(r"\bBREAK\b", text)
    token_chunks = []
    need_word_ids = len(regions) > 0 or (have_advanced_encode and style != "perp")
    for c in chunks:
        c, shuffles = get_function(c.strip(), "(SHIFT|SHUFFLE)", ["0", "default", "default"], return_func_name=True)
        r = c
        for s in shuffles:
            r = shuffle_chunk(s, r)
        if r != c:
            log.info("Shuffled prompt chunk to %s", r)
            c = r
        if sculpts:
            w, method, norm = sculpts[0]
            log.info("Using vector sculptor with method=%s norm=%s w=%s", method, norm, w)
            w = safe_float(w, 1.0)
            t = vector_sculptor_tokens(clip, c, method, norm, w)
        else:
            # Tokenizer returns padded results
            t = clip.tokenize(c, return_word_ids=need_word_ids)
        token_chunks.append(t)
    tokens = token_chunks[0]

    for key in tokens:
        for c in token_chunks[1:]:
            tokens[key].extend(c[key])

    # Non-SDXL has only "l"
    if "g" in tokens and l_prompts:
        text_l = " ".join(l_prompts)
        log.info("Encoded SDXL CLIP_L prompt: %s", text_l)
        tokens["l"] = clip.tokenize(text_l, return_word_ids=need_word_ids)["l"]

    if "g" in tokens and "l" in tokens and len(tokens["l"]) != len(tokens["g"]):
        empty = clip.tokenize("", return_word_ids=need_word_ids)
        while len(tokens["l"]) < len(tokens["g"]):
            tokens["l"] += empty["l"]
        while len(tokens["l"]) > len(tokens["g"]):
            tokens["g"] += empty["g"]

    tokens = fix_word_ids(tokens)

    if len(regions) > 0:
        return encode_regions(clip, tokens, regions, style, normalization)

    if style == "perp":
        if normalization != "none":
            log.warning("Normalization is not supported with perp style weighting. Ignored '%s'", normalization)
        return perp_encode(clip, tokens)

    if "t5xxl" not in tokens and have_advanced_encode and not sculpts:
        if "g" in tokens:
            embs_l = None
            embs_g = None
            pooled = None
            if "l" in tokens:
                embs_l, _ = advanced_encode_from_tokens(
                    tokens["l"],
                    normalization,
                    style,
                    lambda x: encode_token_weights(clip, x, encode_token_weights_l),
                    return_pooled=False,
                )
            if "g" in tokens:
                embs_g, pooled = advanced_encode_from_tokens(
                    tokens["g"],
                    normalization,
                    style,
                    lambda x: encode_token_weights(clip, x, encode_token_weights_g),
                    return_pooled=True,
                    apply_to_pooled=False,
                )
            # Hardcoded clip_balance
            return prepareXL(embs_l, embs_g, pooled, 0.5)
        return advanced_encode_from_tokens(
            tokens["l"],
            normalization,
            style,
            lambda x: clip.encode_from_tokens({"l": x}, return_pooled=True),
            return_pooled=True,
            apply_to_pooled=True,
        )
    else:
        return clip.encode_from_tokens(tokens, return_pooled=True)


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


def do_encode(clip, text, defaults, masks):
    # First style modifier applies to ANDed prompts too unless overridden
    style, normalization, text = get_style(text)
    text, mask_size = get_mask_size(text, defaults)

    # Don't sum ANDs if this is in prompt
    alt_method = "COMFYAND()" in text
    text = text.replace("COMFYAND()", "")

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
    res = []
    scale = sum(abs(weight(p)[0]) for p in prompts if not ("AREA(" in p or "MASK(" in p))
    for prompt in prompts:
        prompt, mask, mask_weight = get_mask(prompt, mask_size, masks)
        w, opts, prompt = weight(prompt)
        text, noise_w, generator = get_noise(text)
        if not w:
            continue
        prompt, area = get_area(prompt)
        prompt, local_sdxl_opts = get_sdxl(prompt, defaults)
        cond, pooled = encode_prompt(clip, prompt, style, normalization)
        cond = apply_noise(cond, noise_w, generator)
        pooled = apply_noise(pooled, noise_w, generator)

        settings = {"prompt": prompt}
        if alt_method:
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

        if mask is not None or area or alt_method or local_sdxl_opts:
            if pooled is not None:
                settings["pooled_output"] = pooled
            conds.append([cond, settings])
        else:
            s = opts.get("scale", scale)
            res.append((cond, pooled, w / s))

    sumconds = [r[0] * r[2] for r in res]
    pooleds = [r[1] for r in res if r[1] is not None]

    if len(res) > 0:
        opts = sdxl_opts
        if pooleds:
            opts["pooled_output"] = sum(equalize(*pooleds))
        sumcond = sum(equalize(*sumconds))
        conds.append([sumcond, opts])
    return conds


def debug_conds(conds):
    r = []
    for i, c in enumerate(conds):
        x = c[1].copy()
        if "pooled_output" in x:
            del x["pooled_output"]
        r.append((i, x))
    return r


def control_to_clip_common(clip, schedules, lora_cache=None, cond_cache=None):
    orig_clip = clip.clone()
    current_loras = {}
    if lora_cache is None:
        lora_cache = {}
    start_pct = 0.0
    conds = []
    cond_cache = cond_cache if cond_cache is not None else {}

    def c_str(c):
        r = [c["prompt"]]
        loras = c["loras"]
        for k in sorted(loras.keys()):
            r.append(k)
            r.append(loras[k]["weight_clip"])
            for lbw, val in loras[k].get("lbw", {}).items():
                r.append(lbw)
                r.append(val)
        return "".join(str(i) for i in r)

    def encode(c):
        nonlocal clip
        nonlocal current_loras
        prompt = c["prompt"]
        loras = c["loras"]
        cachekey = c_str(c)
        cond = cond_cache.get(cachekey)
        if cond is None:
            if loras != current_loras:
                _, clip = apply_loras_from_spec(loras, clip=orig_clip, cache=lora_cache, applied_loras=current_loras)
                current_loras = loras
            cond_cache[cachekey] = do_encode(clip, prompt, schedules.defaults, schedules.masks)
        return cond_cache[cachekey]

    for end_pct, c in schedules:
        interpolations = [
            i
            for i in schedules.interpolations
            if (start_pct >= i[0][0] and start_pct < i[0][-1]) or (end_pct > i[0][0] and start_pct < i[0][-1])
        ]
        new_start_pct = start_pct
        if interpolations:
            min_step = min(i[1] for i in interpolations)
            for i in interpolations:
                control_points, _ = i
                interpolation_end_pct = min(control_points[-1], end_pct)
                interpolation_start_pct = max(control_points[0], start_pct)

                control_points = get_control_points(schedules, control_points, encode)
                cs = linear_interpolator(control_points, min_step, interpolation_start_pct, interpolation_end_pct)
                conds.extend(cs)
                new_start_pct = max(new_start_pct, interpolation_end_pct)
        start_pct = new_start_pct

        if start_pct < end_pct:
            cond = encode(c)
            # Node functions return lists of cond
            cond = conditioning_set_values(
                cond, {"start_percent": round(start_pct, 2), "end_percent": round(end_pct, 2), "prompt": c["prompt"]}
            )
            conds.extend(cond)

        start_pct = end_pct
        log.debug("Conds at the end: %s", debug_conds(conds))

    log.debug("Final cond info: %s", debug_conds(conds))
    return conds
