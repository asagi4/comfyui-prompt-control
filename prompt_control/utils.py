import lark
from pathlib import Path

import logging

log = logging.getLogger('comfyui-prompt-scheduling')
logging.basicConfig()
log.setLevel(logging.INFO)

import time

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.sd
import folder_paths

prompt_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | plain | loraspec | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" WHITESPACE? NUMBER "]"
loraspec: "<lora:" plain (":" WHITESPACE? NUMBER)~1..2 ">"
WHITESPACE: /\s+/
plain: /([^<>\\\[\]():]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")

def getlogger():
    return log

def flatten(x):
    if type(x) in [str, tuple]:
        yield x
    else:
        for g in x:
            yield from flatten(g)

def parse_prompt_schedules(prompt):
    try:
        tree = prompt_parser.parse(prompt)
    except lark.exceptions.LarkError as e:
        log.error("Prompt editing parse error: %s", e)
        return [[1.0, {"prompt": prompt, "loras": []}]]

    def collect(tree):
        res = [1.0]
        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                # Last element in []
                tree.children[-1] = max(min(1.0, float(tree.children[-1])), 0.0)
                res.append(tree.children[-1])

        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, when = args
                yield before or () if step <= when else after

            def start(self, args):
                prompt = []
                loraspecs = []
                args = flatten(args)
                for a in args:
                    if type(a) == str:
                        prompt.append(a)
                    elif a:
                        loraspecs.append(a)
                return {"prompt": "".join(prompt), "loras": loraspecs}

            def plain(self, args):
                yield args[0].value

            def loraspec(self, args):
                name = ''.join(flatten(args[0]))
                params = [float(p) for p in args[1:]]
                return name, params

            def __default__(self, data, children, meta):
                for child in children:
                    yield child

        return AtStep().transform(tree)

    parsed = [[t, at_step(t, tree)] for t in collect(tree)]
    return parsed

def load_loras_from_schedule(schedules, loaded_loras):
    lora_specs = []
    for step, sched in schedules:
        if sched["loras"]:
            lora_specs.extend(sched["loras"])
    loaded_loras = load_loras(lora_specs)
    return loaded_loras


class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        log.debug(f"Executed {self.name} in {elapsed} seconds")

def schedule_for_step(total_steps, step, schedules):
    for end, s in schedules:
        if end*total_steps > step:
            return [end, s]
    return schedules[-1]

def load_loras(lora_specs, loaded_loras=None):
    loaded_loras = loaded_loras or {}
    filenames = [Path(f) for f in folder_paths.get_filename_list("loras")]
    names = set(name for name, _ in lora_specs)
    for name in names:
        if name in loaded_loras:
            continue
        found = False
        for f in filenames:
            if f.stem == name:
                full_path = folder_paths.get_full_path("loras", str(f))
                loaded_loras[name] = comfy.utils.load_torch_file(full_path, safe_load=True)
                found = True
                break
        if not found:
            log.warning("Lora %s not found", name)
    return loaded_loras
