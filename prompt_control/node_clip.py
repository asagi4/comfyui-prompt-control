from . import utils as utils
from .parser import parse_prompt_schedules, parse_cuts
from .utils import Timer, equalize, safe_float
from .perp_weight import perp_encode

import logging
import re

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

AVAILABLE_STYLES.append("perp")
log.info("Use STYLE:weight_interpretation:normalization at the start of a prompt to use advanced encodings")
log.info("Weight interpretations available: %s", ",".join(AVAILABLE_STYLES))
log.info("Normalization types available: %s", ",".join(AVAILABLE_NORMALIZATIONS))


def linear_interpolate_cond(
    start, end, from_step=0.0, to_step=1.0, step=0.1, start_at=None, end_at=None, prompt_start="N/A", prompt_end="N/A"
):
    from_cond, to_cond = equalize(start[0][0], end[0][0])
    from_pooled = start[0][1].get("pooled_output")
    to_pooled = end[0][1].get("pooled_output")
    res = []
    start_at = start_at if start_at is not None else from_step
    end_at = end_at if end_at is not None else to_step
    total_steps = int(round((to_step - from_step) / step, 0))
    num_steps = int(round((end_at - from_step) / step, 0))
    start_on = int(round((start_at - from_step) / step, 0))
    start_pct = start_at
    log.debug(
        f"interpolate_cond {from_step=} {to_step=} {start_at=} {end_at=} {total_steps=} {num_steps=} {start_on=} {step=}"
    )
    x = 1 / (num_steps + 1)
    for s in range(start_on, num_steps + 1):
        factor = round(s * x, 2)
        new_cond = from_cond + (to_cond - from_cond) * factor
        if from_pooled is not None and to_pooled is not None:
            from_pooled, to_pooled = equalize(from_pooled, to_pooled)
            new_pooled = from_pooled + (to_pooled - from_pooled) * factor
        elif from_pooled is not None:
            new_pooled = from_pooled

        n = [new_cond, start[0][1].copy()]
        n[1]["pooled_output"] = new_pooled
        n[1]["start_percent"] = round(1.0 - start_pct, 2)
        n[1]["end_percent"] = max(round(1.0 - (start_pct + step), 2), 0)
        start_pct += step
        start_pct = round(start_pct, 2)
        if prompt_start:
            n[1]["prompt"] = f"linear:{1.0 - factor} / {factor}"
        log.debug(
            "Interpolating at step %s with factor %s (%s, %s)...",
            s,
            factor,
            n[1]["start_percent"],
            n[1]["end_percent"],
        )
        res.append(n)
    if res:
        res[-1][1]["end_percent"] = round(1.0 - end_at, 2)
    return res


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


class CondLinearInterpolate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"start": ("CONDITIONING",), "end": ("CONDITIONING",)},
            "optional": {
                "until": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "step": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "promptcontrol/exp"

    def apply(self, start, end, until=1.0, step=0.1):
        res = linear_interpolate_cond(start, end, 0.0, until, step)
        return (res,)


class ScheduleToCond:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip": ("CLIP",), "prompt_schedule": ("PROMPT_SCHEDULE",)}}

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, clip, prompt_schedule):
        with Timer("ScheduleToCond"):
            r = (control_to_clip_common(self, clip, prompt_schedule),)
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
    CATEGORY = "promptcontrol/old"
    FUNCTION = "parse"

    def parse(self, clip, text, filter_tags=""):
        parsed = parse_prompt_schedules(text).with_filters(filter_tags)
        return (control_to_clip_common(self, clip, parsed),)


def get_style(text, default_style="comfy", default_normalization="none"):
    style = default_style
    normalization = default_normalization
    text = text.strip()
    if text.startswith("STYLE:"):
        style, text = text.split(maxsplit=1)
        r = style.split(":", maxsplit=2)
        style = r[1]
        if len(r) > 2:
            normalization = r[2]
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


def encode_prompt(clip, text, default_style="comfy", default_normalization="none"):
    style, normalization, text = get_style(text, default_style, default_normalization)
    text, regions = parse_cuts(text)
    chunks = text.split("BREAK")
    token_chunks = []
    for c in chunks:
        if not c.strip():
            continue
        # Tokenizer returns padded results
        token_chunks.append(clip.tokenize(c, return_word_ids=have_advanced_encode and style != "perp"))
    tokens = token_chunks[0]
    for c in token_chunks[1:]:
        if isinstance(tokens, list):
            tokens.extend(c)
        else:
            # dict, SDXL
            for key in tokens:
                tokens[key].extend(c[key])

    if len(regions) > 0:
        return encode_regions(clip, tokens, regions, style, normalization)

    if style == "perp":
        if normalization != "none":
            log.warning("Normalization is not supported with perp style weighting. Ignored '%s'", normalization)
        return perp_encode(clip, tokens)

    if have_advanced_encode:
        if isinstance(tokens, dict):
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
                    apply_to_pooled=True,
                )
            # Hardcoded clip_balance
            return prepareXL(embs_l, embs_g, pooled, 0.5)
        return advanced_encode_from_tokens(
            tokens,
            normalization,
            style,
            lambda x: clip.encode_from_tokens(x, return_pooled=True),
            return_pooled=True,
            apply_to_pooled=True,
        )
    else:
        return clip.encode_from_tokens(tokens, return_pooled=True)


def get_area(text):
    rex = r"AREA\((.*?)\)"
    areas = re.findall(rex, text)
    if not areas:
        return text, None
    text = re.sub(rex, "", text)
    area = areas[0]
    args = [1.0, 1.0, 0, 0, 1.0]
    for i, arg in list(enumerate(area.split(",")))[:5]:
        args[i] = safe_float(arg, args[i])
    h, w, y, x, strength = args
    return text, (("percentage", h, w, y, x), strength)


def do_encode(clip, text):
    # First style modifier applies to ANDed prompts too unless overridden
    style, normalization, text = get_style(text)
    prompts = [p.strip() for p in text.split("AND") if p.strip()]
    if len(prompts) == 1:
        cond, pooled = encode_prompt(clip, prompts[0], style, normalization)
        return [[cond, {"pooled_output": pooled}]]

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
    scale = sum(abs(weight(p)[0]) for p in prompts if not "AREA(" in p)
    for prompt in prompts:
        w, opts, prompt = weight(prompt)
        if not w:
            continue
        prompt, area = get_area(prompt)
        cond, pooled = encode_prompt(clip, prompt, style, normalization)
        if area:
            log.info("Prompt %s with area %s", prompt, area)
            conds.append(
                [cond, {"pooled_output": pooled, "area": area[0], "strength": area[1], "set_area_to_bounds": False}]
            )
        else:
            s = opts.get("scale", scale)
            res.append((cond, pooled, w / s))

    sumconds = [r[0] * r[2] for r in res]
    pooleds = [r[1] for r in res if r is not None]

    if len(res) > 0:
        conds.append(
            [sum(equalize(*sumconds)), {"pooled_output": sum(equalize(*pooleds)) if len(pooleds) > 0 else None}]
        )
    return conds


def debug_conds(conds):
    r = []
    for i, c in enumerate(conds):
        x = c[1].copy()
        del x["pooled_output"]
        r.append((i, x))
    return r


def control_to_clip_common(self, clip, schedules, lora_cache=None, cond_cache=None):
    orig_clip = clip.clone()
    current_loras = {}
    loaded_loras = schedules.load_loras(lora_cache)
    start_pct = 0.0
    conds = []
    cond_cache = cond_cache if cond_cache is not None else {}
    use_static_steps = False

    def load_clip_lora(clip, loraspec):
        if not loraspec:
            return clip
        key_map = utils.get_lora_keymap(clip=clip)
        for name, params in loraspec.items():
            if name not in loaded_loras:
                log.warn("%s not loaded, skipping", name)
                continue
            if params["weight_clip"] == 0:
                continue
            clip = utils.load_lora(clip, loaded_loras[name], params["weight_clip"], key_map, clone=False)
            log.info("CLIP LoRA applied: %s:%s", name, params["weight_clip"])
        return clip

    def c_str(c):
        r = [c["prompt"]]
        loras = c["loras"]
        for k in sorted(loras.keys()):
            r.append(k)
            r.append(loras[k]["weight_clip"])
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
                clip = load_clip_lora(orig_clip.clone(), loras)
                current_loras = loras
            cond_cache[cachekey] = do_encode(clip, prompt)
        return cond_cache[cachekey]

    for end_pct, c in schedules:
        if "#ABS#" in c["prompt"]:
            use_static_steps = True
            log.info("Using static schedules")
            c["prompt"] = c["prompt"].replace("#ABS#", "")
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
            for n in cond:
                n = [n[0], n[1].copy()]
                n[1]["start_percent"] = round(1.0 - start_pct, 2)
                n[1]["end_percent"] = round(1.0 - end_pct, 2)
                n[1]["prompt"] = c["prompt"]
                conds.append(n)

        start_pct = end_pct
        log.debug("Conds at the end: %s", debug_conds(conds))

    log.debug("Final cond info: %s", debug_conds(conds))
    if use_static_steps:
        for t in range(len(conds)):
            conds[t][1]["absolute_timesteps"] = True
    return conds
