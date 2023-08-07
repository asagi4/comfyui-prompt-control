from . import utils as utils
from .parser import parse_prompt_schedules

from nodes import NODE_CLASS_MAPPINGS as COMFY_NODES

import logging

log = logging.getLogger("comfyui-prompt-control")


def linear_interpolate_cond(
    start, end, start_step=0.0, until_step=1.0, step=0.1, total_step=None, prompt_start="N/A", prompt_end="N/A"
):
    from_cond = start[0][0]
    to_cond = end[0][0]
    from_pooled = start[0][1].get("pooled_output")
    to_pooled = end[0][1].get("pooled_output")
    res = []
    num_steps = int((until_step - start_step) / step)
    start_pct = start_step
    for s in range(1, num_steps + 1):
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
    res[-1][1]["end_percent"] = round(1.0 - until_step, 2)
    return res


def linear_interpolate(schedule, from_step, to_step, step, encode):
    start_prompt = schedule.at_step(from_step)
    # Returns the prompt to interpolate towards
    conds = []
    end_at = 0
    while end_at < to_step:
        r = schedule.interpolation_at(from_step)
        log.debug("Interpolation target: %s", r)
        if not r:
            raise Exception("No interpolation target? This isn't supposed to happen")
        end_at, end_prompt = r
        start = encode(start_prompt)
        end = encode(end_prompt)
        log.info("Interpolating %s to %s, (%s, %s, %s)", start_prompt, end_prompt, from_step, to_step, step)
        cs = linear_interpolate_cond(
            start, end, from_step, end_at, step, to_step, start_prompt[1]["prompt"], end_prompt[1]["prompt"]
        )
        conds.extend(cs)
        from_step = end_at
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
        res = linear_interpolate_cond(start, end, 0.0, until, step, lambda x: x)
        return (res,)


class ScheduleToCond:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip": ("CLIP",), "prompt_schedule": ("PROMPT_SCHEDULE",)}}

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, clip, prompt_schedule):
        return (control_to_clip_common(self, clip, prompt_schedule),)


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


def do_encode(that, clip, text):
    def fallback():
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    # Super hacky way to call other nodes
    # <f.Nodename(param=a,param2=b)>
    if text.startswith("<f."):
        encodernode, text = text[3:].split(">", 1)
        encoderparams = {}
        paramstart = encodernode.find("(")
        paramend = encodernode.find(")")
        if paramstart > 0 and paramend > paramstart:
            ps = encodernode[paramstart + 1 : paramend]
            encodernode = encodernode[:paramstart]
            for p in ps.split(","):
                k, v = p.split("=", 1)
                encoderparams[k.strip().lower()] = v.strip()

        node = COMFY_NODES.get(encodernode)
        if not node or "CONDITIONING" not in node.RETURN_TYPES:
            log.error("Invalid encoder node: %s, ignoring", encodernode)
            return fallback()
        ret_index = node.RETURN_TYPES.index("CONDITIONING")
        log.info("Attempting to use %s", encodernode)
        input_types = node.INPUT_TYPES()
        r = input_types["required"]
        params = {}
        for k in r:
            t = r[k][0]
            if t == "STRING":
                params[k] = text
                log.info("Set %s=%s", k, params[k])
            elif t == "CLIP":
                params[k] = clip
                log.info("Set %s to the CLIP model", k)
            elif t in ("INT", "FLOAT"):
                f = __builtins__[t.lower()]
                if k in encoderparams:
                    params[k] = f(encoderparams[k])
                else:
                    params[k] = r[k][1]["default"]
                log.info("Set %s=%s", k, params[k])
            elif isinstance(t, list):
                if k in encoderparams and k in t:
                    params[k] = encoderparams[k]
                else:
                    params[k] = t[0]
                log.info("Set %s=%s", k, params[k])
            nodefunc = getattr(node, node.FUNCTION)
        res = nodefunc(that, **params)[ret_index]
        return res
    return fallback()


def debug_conds(conds):
    r = []
    for i, c in enumerate(conds):
        x = c[1].copy()
        del x["pooled_output"]
        r.append((i, x))
    return r


def control_to_clip_common(self, clip, schedules, lora_cache=None):
    orig_clip = clip.clone()
    current_loras = {}
    loaded_loras = schedules.load_loras(lora_cache)
    start_pct = 0.0
    conds = []
    cond_cache = {}

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
            return do_encode(self, clip, x[1]["prompt"])

        if interpolations:
            start_step, end_step, step = interpolations
            start_pct = start_step
            cs = linear_interpolate(schedules, start_step, end_step, step, encode)
            conds.extend(cs)
        else:
            cond = do_encode(self, clip, prompt)
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
