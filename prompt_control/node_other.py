import logging
from .parser import parse_prompt_schedules
from .utils import steps
import math
from os import environ
import random
import re
from pathlib import Path
from datetime import datetime
from server import PromptServer

log = logging.getLogger("comfyui-prompt-control")


def template(template, sequence, *funcs):
    funcs = [lambda x: x] + list(*funcs)
    res = []
    for item in sequence:
        x = template
        for i, f in enumerate(funcs):
            x = x.replace(f"${i}", str(f(i)))
        res.append(x)

    return "".join(res)


def wildcard_prompt_handler(json_data):
    log.info("Resolving wildcards...")
    for node_id in json_data["prompt"].keys():
        if json_data["prompt"][node_id]["class_type"] == "SimpleWildcard":
            handle_wildcard_node(json_data, node_id)
    return json_data


def handle_wildcard_node(json_data, node_id):
    wildcard_info = json_data.get("extra_data", {}).get("extra_pnginfo", {}).get("SimpleWildcard", {})
    n = json_data["prompt"][node_id]
    if not (n["inputs"].get("use_pnginfo") and node_id in wildcard_info):
        text = SimpleWildcard.select(n["inputs"]["text"], n["inputs"]["seed"])

    if text.strip() != n["inputs"]["text"].strip():
        json_data["prompt"][node_id]["inputs"]["use_pnginfo"] = True
        wildcard_info[node_id] = text
        json_data["extra_data"]["extra_pnginfo"]["SimpleWildcard"] = wildcard_info
    return json_data


PromptServer.instance.add_on_prompt_handler(wildcard_prompt_handler)


def variable_substitution(text):
    var_re = re.compile(r"(\$[a-z]+)\s*=([^;\n]*);?")
    m = var_re.search(text)
    while m:
        var = m[1]
        sub = m[2]
        s, e = m.span()
        text = text[:s] + text[e:]
        log.info("Substituting %s with '%s'", var, sub)
        text = text.replace(var, sub)
        m = var_re.search(text)
    return text


class SimpleWildcard:
    RAND = random.Random()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {"use_pnginfo": ("BOOLEAN", {"default": False})},
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)

    CATEGORY = "promptcontrol/tools"
    FUNCTION = "doit"

    @classmethod
    def read_wildcards(cls, name):
        path = environ.get("PC_WILDCARD_BASEDIR", "wildcards")

        f = (Path(path) / Path(name)).with_suffix(".txt")
        try:
            with open(f, "r") as file:
                return [l.strip() for l in file.readlines() if l.strip()]
        except:
            return [name]

    @classmethod
    def select(cls, text, seed):
        cls.RAND.seed(seed)
        matches = re.findall(r"(\$([A-Za-z0-9_/.-]+)(\+[0-9]+)?\$)", text)
        for placeholder, wildcard, offset in matches:
            if offset:
                offset = int(offset[1:])
                cls.RAND.seed(seed + offset)
            w = cls.RAND.choice(cls.read_wildcards(wildcard))
            text = text.replace(placeholder, w, 1)
            log.info("Selected wildcard %s for %s", w, placeholder)
            if offset:
                cls.RAND.seed(seed)
        return text

    def doit(self, text, seed, extra_pnginfo, unique_id, use_pnginfo=False):
        if use_pnginfo and unique_id in extra_pnginfo.get("SimpleWildcard", {}):
            text = extra_pnginfo["SimpleWildcard"][unique_id]
            log.info("SimpleWildcard using prompt: %s", text)
        text = variable_substitution(text)
        return (text,)


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


class FilterSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"prompt_schedule": ("PROMPT_SCHEDULE",)},
            "optional": {
                "tags": ("STRING", {"default": ""}),
                "start": ("FLOAT", {"min": 0.00, "max": 1.00, "default": 0.0, "step": 0.01}),
                "end": ("FLOAT", {"min": 0.00, "max": 1.00, "default": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("PROMPT_SCHEDULE",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, prompt_schedule, tags="", start=0.0, end=1.0):
        p = prompt_schedule.with_filters(tags, start=start, end=end)
        log.debug(
            f"Filtered {prompt_schedule.parsed_prompt} with: ({tags}, {start}, {end}); the result is %s",
            p.parsed_prompt,
        )
        return (p,)


class PromptToSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("PROMPT_SCHEDULE",)
    CATEGORY = "promptcontrol"
    FUNCTION = "parse"

    def parse(self, text, filter_tags=""):
        schedules = parse_prompt_schedules(text)
        return (schedules,)


def clamp(a, b, c):
    return max(a, min(b, c))


JINJA_ENV = {
    "pi": math.pi,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
    "abs": abs,
    "clamp": clamp,
    "round": round,
    "template": template,
    "steps": steps,
    "datetime": datetime,
}

for fname in ["sqrt", "sin", "cos", "tan", "asin", "acos", "atan"]:
    f = getattr(math, fname)
    JINJA_ENV[fname] = lambda x: round(f(x), 2)


def render_jinja(text):
    from jinja2 import Environment

    jenv = Environment(
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<=",
        variable_end_string="=>",
        comment_start_string="<#",
        comment_end_string="#>",
    )

    return jenv.from_string(text, globals=JINJA_ENV).render()


class JinjaRender:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"default": "", "multiline": True})}}

    RETURN_TYPES = ("STRING",)

    CATEGORY = "promptcontrol/tools"
    FUNCTION = "render"

    def render(self, text):
        t = render_jinja(text)
        if t.strip() != text.strip():
            log.info("Jinja render result: %s", re.sub("\s+", " ", t, flags=re.MULTILINE))
        return (t,)


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
