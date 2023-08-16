from . import utils as utils
from .parser import parse_prompt_schedules
from .utils import Timer

import logging
import re

from math import lcm

log = logging.getLogger("comfyui-prompt-control")


def linear_interpolate_cond(
    start, end, from_step=0.0, to_step=1.0, step=0.1, start_at=None, end_at=None, prompt_start="N/A", prompt_end="N/A"
):
    from_cond = start[0][0]
    to_cond = end[0][0]
    from_pooled = start[0][1].get("pooled_output")
    to_pooled = end[0][1].get("pooled_output")
    res = []
    start_at = start_at or from_step
    end_at = end_at or to_step
    num_steps = int((to_step - from_step) / step)
    start_on = int((start_at - from_step) / step) + 1
    end_on = int((end_at - from_step) / step) + 1
    start_pct = start_at
    log.debug(
        f"interpolate_cond {from_step=} {to_step=} {start_at=} {end_at=} {num_steps=} {start_on=} {end_on=} {step=}"
    )
    for s in range(start_on, end_on):
        factor = round(s / (num_steps + 1), 2)
        new_cond = from_cond + (to_cond - from_cond) * factor
        if from_pooled is not None and to_pooled is not None:
            new_pooled = from_pooled + (to_pooled - from_pooled) * factor
        elif from_pooled is not None:
            new_pooled = from_pooled

        n = [new_cond, start[0][1].copy()]
        n[1]["pooled_output"] = new_pooled
        n[1]["start_percent"] = round(1.0 - start_pct, 2)
        n[1]["end_percent"] = round(1.0 - (start_pct + step), 2)
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


def linear_interpolate(schedule, from_step, to_step, step, start_at, encode):
    start_prompt = schedule.at_step(start_at)
    # Returns the prompt to interpolate towards
    conds = []
    end_at = start_at
    start = encode(start_prompt)
    while end_at < to_step:
        r = schedule.interpolation_at(start_at)
        log.debug("Interpolation target: %s", r)
        if not r:
            log.info("No interpolation target?")
            return conds
        end_at, end_prompt = r
        end_at = min(to_step, end_at)
        end = encode(end_prompt)
        log.info(
            "Interpolating %s to %s, (%s, %s, %s)",
            start_prompt[1]["prompt"],
            end_prompt[1]["prompt"],
            from_step,
            to_step,
            step,
        )
        cs = linear_interpolate_cond(
            start, end, from_step, to_step, step, start_at, end_at, start_prompt[1]["prompt"], end_prompt[1]["prompt"]
        )
        conds.extend(cs)
        start_at = end_at
        start = end
        start_prompt = end_prompt
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


def encode_prompt(clip, text):
    chunks = text.split("BREAK")
    tokens = []
    for c in chunks:
        if not c.strip():
            continue
        # Tokenizer returns padded results
        tokens.extend(clip.tokenize(c))
    return clip.encode_from_tokens(tokens, return_pooled=True)


def do_encode(clip, text):
    prompts = [p.strip() for p in text.split("AND") if p.strip()]
    if len(prompts) == 1:
        cond, pooled = encode_prompt(clip, prompts[0])
        return [[cond, {"pooled_output": pooled}]]

    def weight(t):
        x = re.findall(r":(-?\d\.?\d*)$", t) or [1.0]
        return float(x[0])

    conds = []
    pooleds = []
    max_len = 0
    scale = sum(abs(weight(p)) for p in prompts)
    for prompt in prompts:
        w = weight(prompt)
        if not w:
            continue
        cond, pooled = encode_prompt(clip, prompt)
        if max_len == 0:
            max_len = cond.shape[1]
        else:
            max_len = lcm(max_len, cond.shape[1])
        conds.append(cond * (w / scale))
        pooleds.append(pooled)

    def repeat(x):
        for i in x:
            if i.shape[1] < max_len:
                yield i.repeat(1, max_len // i.shape[1], 1)
            else:
                yield i

    return [[sum(repeat(conds)), {"pooled_output": sum(repeat(pooleds))}]]


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
            log.info("CLIP LoRA loaded: %s:%s", name, params["weight_clip"])
        return clip

    def c_str(c):
        r = [c["prompt"]]
        loras = c["loras"]
        for k in sorted(loras.keys()):
            r.append(k)
            r.append(loras[k]["weight_clip"])
        return "".join(str(i) for i in r)

    max_interpolated = 0.0
    for end_pct, c in schedules:
        log.debug("Encoding at %s: %s", end_pct, c["prompt"])
        prompt = c["prompt"]
        loras = c["loras"]
        cachekey = c_str(c)
        cond = cond_cache.get(cachekey)
        if cond is None:
            if loras != current_loras:
                clip = load_clip_lora(orig_clip.clone(), loras)
                current_loras = loras

        interpolations = c.get("interpolations")

        def encode(x):
            return do_encode(clip, x[1]["prompt"])

        if interpolations:
            start_step, end_step, step = interpolations
            start_pct = max(start_step, max_interpolated)
            max_interpolated = max(end_step, max_interpolated)
            cs = linear_interpolate(schedules, start_step, end_step, step, start_pct, encode)
            conds.extend(cs)
        else:
            with Timer("CLIP Encode"):
                cond = do_encode(clip, prompt)
            cond_cache[cachekey] = cond
            # Node functions return lists of cond
            for n in cond:
                n = [n[0], n[1].copy()]
                n[1]["start_percent"] = round(1.0 - start_pct, 2)
                n[1]["end_percent"] = round(1.0 - end_pct, 2)
                n[1]["prompt"] = prompt
                conds.append(n)

        start_pct = end_pct
        log.debug("Conds at the end: %s", debug_conds(conds))

    log.debug("Final cond info: %s", debug_conds(conds))
    return conds
