import logging
import re
import torch
from .utils import safe_float, get_function, parse_floats, lora_name_to_file
from comfy_extras.nodes_mask import FeatherMask, MaskComposite
import comfy.utils
import comfy.hooks
import folder_paths
from .perp_weight import perp_encode_new

log = logging.getLogger("comfyui-prompt-control")

AVAILABLE_STYLES = ["comfy", "perp"]
AVAILABLE_NORMALIZATIONS = ["none"]

have_advanced_encode = False
try:
    import custom_nodes.ComfyUI_ADV_CLIP_emb.adv_encode as adv_encode

    have_advanced_encode = True
    AVAILABLE_STYLES.extend(["A1111", "compel", "comfy++", "down_weight"])
    AVAILABLE_NORMALIZATIONS.extend(["mean", "length", "length+mean"])
except ImportError:
    pass


class PCLoraHooksFromSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"prompt_schedule": ("PROMPT_SCHEDULE",)},
        }

    RETURN_TYPES = ("HOOKS",)
    OUTPUT_TOOLTIPS = ("set of hooks created from the prompt schedule",)
    CATEGORY = "promptcontrol/_unstable"
    FUNCTION = "apply"

    def apply(self, prompt_schedule):
        return (lora_hooks_from_schedule(prompt_schedule),)


class PCEncodeSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP",), "prompt_schedule": ("PROMPT_SCHEDULE",)},
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol/_unstable"
    FUNCTION = "apply"

    def apply(self, clip, prompt_schedule):
        return (encode_schedule(clip, prompt_schedule),)


class PCEncodeSingle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP",), "prompt": ("STRING", {"multiline": True})},
            "optional": {"defaults": ("SCHEDULE_DEFAULTS",)},
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol/_unstable"
    FUNCTION = "apply"

    def apply(self, clip, prompt, defaults=None):
        return (do_encode(clip, prompt, 0, 1.0, defaults or {}, None),)


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


def encode_prompt(
    clip,
    text,
    settings,
    default_style="comfy",
    default_normalization="none",
    clip_weights=None,
) -> list[tuple[torch.Tensor, dict[str]]]:
    style, normalization, text = get_style(text, default_style, default_normalization)
    clip_weights, text = get_clipweights(text, clip_weights)
    # defaults=None means there is no argument parsing at all
    text, l_prompts = get_function(text, "CLIP_L", defaults=None)
    chunks = re.split(r"\bBREAK\b", text)
    token_chunks = []
    need_word_ids = have_advanced_encode or style == "comfy" and normalization == "none"
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

    tes = []
    empty = clip.tokenize("")
    for k in tokens:
        if k in ["g", "l"]:
            tes.append(f"clip_{k}")
            empty[f"clip_{k}"] = empty[k]
            empty.pop(k)
        else:
            tes.append(k)

    clip = hook_te(clip, tes, style, normalization, clip_weights, empty)

    return clip.encode_from_tokens_scheduled(tokens, add_dict=settings)


def handle_weights(spec, te_name, output):
    if not spec:
        return output

    if te_name.startswith("clip_"):
        te_name = te_name[5:]

    if isinstance(output, tuple):
        out, pooled = output
        if te_name in spec:
            log.info("Weighting %s output by %s", te_name, spec[te_name])
            out = out * spec[te_name]
        pkey = te_name + "_pooled"
        if pkey in spec:
            log.info("Weighting %s pooled output by %s", te_name, spec[pkey])
            pooled = pooled * spec[pkey]

        return out, pooled
    else:
        if te_name in spec:
            log.info("Weighting %s output by %s", te_name, spec[te_name])
            output = output * spec[te_name]
        return output


def make_patch(te_name, orig_fn, normalization, style, clip_weights, empty_tokens):
    def encode(t):
        if style == "perp":
            r = perp_encode_new(orig_fn, t, empty_tokens[te_name])
        else:
            r = adv_encode.advanced_encode_from_tokens(
                t, normalization, style, orig_fn, return_pooled=True, apply_to_pooled=False
            )
        return handle_weights(clip_weights, te_name, r)

    return encode


def hook_te(clip, te_names, style, normalization, clip_weights, empty_tokens):
    if not have_advanced_encode or style == "comfy" and normalization == "none":
        return clip
    newclip = clip.clone()
    for te_name in te_names:
        if hasattr(clip.patcher.model, te_name):
            log.debug("Hooked into %s with style=%s, normalization=%s", te_name, style, normalization)
            newclip.patcher.add_object_patch(
                f"{te_name}.encode_token_weights",
                make_patch(
                    te_name,
                    clip.patcher.get_model_object(f"{te_name}.encode_token_weights"),
                    normalization,
                    style,
                    clip_weights,
                    empty_tokens,
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


def do_encode(clip, text, start_pct, end_pct, defaults, masks):
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
    scale = sum(abs(weight(p)[0]) for p in prompts if not ("AREA(" in p or "MASK(" in p))
    for prompt in prompts:
        prompt, mask, mask_weight = get_mask(prompt, mask_size, masks)
        w, opts, prompt = weight(prompt)
        text, noise_w, generator = get_noise(text)
        if not w:
            continue
        prompt, area = get_area(prompt)
        prompt, local_sdxl_opts = get_sdxl(prompt, defaults)
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
        x = encode_prompt(clip, prompt, settings, style, normalization)
        conds.extend(x)

    return conds


def debug_conds(conds):
    r = []
    for i, c in enumerate(conds):
        x = c[1].copy()
        if "pooled_output" in x:
            del x["pooled_output"]
        r.append((i, x))
    return r


def lora_hooks_from_schedule(schedules):
    start_pct = 0.0

    lora_cache = {}

    all_hooks = []

    prev_loras = {}

    def create_hook(loraspec, start_pct, end_pct):
        nonlocal lora_cache
        hooks = []
        hook_kf = comfy.hooks.HookKeyframeGroup()
        for lora, info in loras.items():
            path = lora_name_to_file(lora)
            if not path:
                continue
            if path not in lora_cache:
                lora_cache[path] = comfy.utils.load_torch_file(
                    folder_paths.get_full_path("loras", path), safe_load=True
                )
            new_hook = comfy.hooks.create_hook_lora(
                lora_cache[path], strength_model=info["weight"], strength_clip=info["weight_clip"]
            )
            # Set hook_ref so that identical hooks compare equal
            new_hook.hooks[0].hook_ref = f"pc-{path}-{info['weight']}-{info['weight_clip']}"
            hooks.append(new_hook)
        if start_pct > 0.0:
            kf = comfy.hooks.HookKeyframe(strength=0.0, start_percent=0.0)
            hook_kf.add(kf)
        kf = comfy.hooks.HookKeyframe(strength=1.0, start_percent=start_pct)
        hook_kf.add(kf)
        if end_pct < 1.0:
            kf = comfy.hooks.HookKeyframe(strength=0.0, start_percent=end_pct)
            hook_kf.add(kf)
        hooks = comfy.hooks.HookGroup.combine_all_hooks(hooks)
        if hooks:
            hooks.set_keyframes_on_hooks(hook_kf=hook_kf)
        return hooks

    consolidated = []

    prev_loras = {}
    for end_pct, c in reversed(list(schedules)):
        loras = c["loras"]
        if loras != prev_loras:
            consolidated.append((end_pct, loras))
        prev_loras = loras
    consolidated = reversed(consolidated)

    for end_pct, loras in consolidated:
        log.info("Creating LoRA hook from %s to %s: %s", start_pct, end_pct, loras)
        hook = create_hook(loras, start_pct, end_pct)
        all_hooks.append(hook)
        start_pct = end_pct

    del lora_cache

    all_hooks = [x for x in all_hooks if x is not None]

    if all_hooks:
        hooks = comfy.hooks.HookGroup.combine_all_hooks(all_hooks)
        return hooks


def encode_schedule(clip, schedules):
    start_pct = 0.0
    conds = []
    for end_pct, c in schedules:
        if start_pct < end_pct:
            prompt = c["prompt"]
            cond = do_encode(clip, prompt, start_pct, end_pct, schedules.defaults, schedules.masks)
            conds.extend(cond)
        start_pct = end_pct

    log.debug("Final cond info: %s", debug_conds(conds))
    return conds
