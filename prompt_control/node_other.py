import logging
from .parser import parse_prompt_schedules

log = logging.getLogger("comfyui-prompt-control")


def steps(start, end=None, step=0.1):
    if end is None:
        end = start
        start = step
    while start <= end:
        yield start
        start += step
        start = round(start, 2)


class StringConcat:
    @classmethod
    def INPUT_TYPES(s):
        t = ("STRING", {"default": ""})
        return {
            "optional": {
                "string1": t,
                "string2": t,
                "string3": t,
                "string4": t,
            }
        }

    RETURN_TYPES = ("STRING",)

    CATEGORY = "promptcontrol/tools"
    FUNCTION = "cat"

    def cat(self, string1="", string2="", string3="", string4=""):
        return string1 + string2 + string3 + string4


def filter_schedules(cutoff, schedules):
    return [(t, i) for t, i in schedules if t >= cutoff]


class PromptToSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "cutoff": ("FLOAT", {"min": 0.00, "max": 1.00, "default": 0.0, "step": 0.01}),
                "filter_tags": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("PROMPT_SCHEDULE",)
    CATEGORY = "promptcontrol"
    FUNCTION = "parse"

    def parse(self, text, cutoff=0.0, filter_tags=""):
        schedules = filter_schedules(cutoff, parse_prompt_schedules(text, filter_tags))
        return (schedules,)


class JinjaRender:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"default": "", "multiline": True})}}

    RETURN_TYPES = ("STRING",)

    CATEGORY = "promptcontrol/tools"
    FUNCTION = "render"

    def render(self, text):
        from jinja2 import Environment
        import math

        jenv = Environment(
            block_start_string="<%",
            block_end_string="%>",
            variable_start_string="<=",
            variable_end_string="=>",
            comment_start_string="<#",
            comment_end_string="#>",
        )

        funcs = dict(m=math, steps=steps, min=min, max=max, abs=abs)
        return jenv.from_string(text, globals=funcs).render()


class ConditioningCutoff:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conds": ("CONDITIONING",),
                "cutoff": ("FLOAT", {"min": 0.00, "max": 1.00, "default": 0.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol/tools"
    FUNCTION = "apply"

    def apply(self, conds, cutoff):
        res = []
        new_start = 1.0
        for c in conds:
            end = c[1].get("end_percent", 0.0)
            if 1.0 - end < cutoff:
                log.debug("Chose to remove prompt '%s'", c[1].get("prompt", "N/A"))
                continue
            c = [c[0].clone(), c[1].copy()]
            c[1]["start_percent"] = new_start
            c[1]["end_percent"] = end
            new_start = end
            res.append(c)

        log.debug("Conds after filter: %s", [(c[1]["prompt"], c[1]["start_percent"], c[1]["end_percent"]) for c in res])
        return (res,)
