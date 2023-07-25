import sys
import os
import lark
from nodes import common_ksampler
from pathlib import Path

import logging

log = logging.getLogger('comfyui-prompt-scheduling')
logging.basicConfig()
log.setLevel(logging.INFO)

import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.sample

if not getattr(comfy.sample.sample, 'prompt_control_monkeypatch', False):
    print("Monkeypatching comfy.sample.sample to support callbacks")
    orig_sample = comfy.sample.sample
    def sample(*args, **kwargs):
        model = args[0]
        if hasattr(model, 'prompt_control_callback'):
            args, kwargs = model.prompt_control_callback(*args, **kwargs)
        return orig_sample(*args, **kwargs)
    setattr(sample, 'prompt_control_monkeypatch', True)
    comfy.sample.sample = sample


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
loraspec: "<" plain (":" WHITESPACE? NUMBER)~1..2 ">"
WHITESPACE: /\s+/
plain: /([^<>\\\[\]():]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")



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

class LoRAScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "clip": ("CLIP",),
                 "text": ("STRING", {"multiline": True}),
                }
               }
    RETURN_TYPES = ('MODEL',)
    CATEGORY = 'nodes'
    FUNCTION = 'apply'

    def __init__(self):
        self.orig_model = None
        self.orig_text = None

    def apply(self, model, clip, text):
        model = model.clone()
        def sampler_cb(*args, **kwargs):
            state = {}
            # Store the original in case we need to reset
            # We need this for LoRA keys
            state['clip'] = clip
            state['orig_model'] = args[0]
            state['model'] = args[0]
            state['steps'] = args[2] # steps
            state['current_step'] = kwargs['start_step'] or 1
            state['last_step'] = min(state['steps'], kwargs['last_step'] or state['steps'])
            if not 'applied_loras' in state:
                state['applied_loras'] = []

            setattr(model, 'prompt_control_state', state)
            
            return (args, kwargs)

        setattr(model, 'prompt_control_callback', sampler_cb)


        def noop(model_fn, params):
            return model_fn(params['input'], params['timestep'], **params['c'])
        orig_model_fn = model.model_options.get('model_function_wrapper', noop)



        schedules = parse_prompt_schedules(text)
        loaded_loras = {}
        loaded_loras = load_loras_from_schedule(schedules, loaded_loras)

        def unet_wrapper(model_fn, params):
            s = model.prompt_control_state
            sched = schedule_for_step(s['steps'], s['current_step'], schedules)
            lora_spec = sorted(sched[1]['loras'])
                
            if s['applied_loras'] != lora_spec:
                apply_loras(s, lora_spec, loaded_loras)
                s['applied_loras'] = lora_spec
            s['current_step'] += 1

            if not orig_model_fn:
                r = model_fn(params['input'], params['timestep'], **params['c'])
            else:
                r = orig_model_fn(model_fn, params)

            if s['current_step'] > s['last_step']:
                if s["applied_loras"]:
                    log.info("sampling done, unpatching model")
                    s["model"].unpatch_model()
            return r

        model.set_model_unet_function_wrapper(unet_wrapper)

        return (model,)

class EditableCLIPEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"clip": ("CLIP",),
                 "model": ("MODEL",),
                 "text": ("STRING", {"multiline": True}),
                }
               }
    RETURN_TYPES = ('CONDITIONING',)
    CATEGORY = 'nodes'
    FUNCTION = 'parse'

    def __init__(self):
        self.loaded_loras = {}
        self.current_loras = []
        self.orig_clip = None

    def load_clip_lora(self, clip, model, loraspec):
        if not loraspec:
            return clip
        key_map = comfy.sd.model_lora_keys_unet(model.model)
        key_map = comfy.sd.model_lora_keys_clip(clip.cond_stage_model, key_map)
        if self.current_loras != loraspec:
            clip = clip.clone()
            for l in loraspec:
                name, w = l
                w = w + [w[0]]
                if name not in self.loaded_loras:
                    log.warn("%s not loaded, skipping", name)
                    continue
                loaded = comfy.sd.load_lora(self.loaded_loras[name], key_map)
                clip.add_patches(loaded, w[1])
                log.info("CLIP LoRA loaded: %s:%s", name, w[1])
        return clip

    def parse(self, clip, model, text):
        parsed = parse_prompt_schedules(text)
        self.current_loras = []
        self.loaded_loras = load_loras_from_schedule(parsed, self.loaded_loras)
        self.orig_clip = clip
        start_pct = 0.0
        conds = []
        for end_pct, c in parsed:
            if c["loras"] != self.current_loras:
                clip = self.load_clip_lora(clip, model, c["loras"])
                self.current_loras = c["loras"]
            conds.append([clip.encode(c["prompt"]), {"start_percent": 1.0 - start_pct, "end_percent": 1.0 - end_pct}])
            start_pct = end_pct
        return (conds,)


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

def apply_loras(s, lora_specs, loaded_loras):
    model = s['model']
    clip = s['clip']
    key_map = comfy.sd.model_lora_keys_unet(model.model)
    key_map = comfy.sd.model_lora_keys_clip(clip.cond_stage_model, key_map)

    with Timer("Unpatch model"):
        model.unpatch_model()
        model = s['orig_model'].clone()

    for name, weights in lora_specs:
        if name not in loaded_loras:
            continue
        loaded = comfy.sd.load_lora(loaded_loras[name], key_map)
        model = model.clone()
        model.add_patches(loaded, weights[0])
        log.info("Loaded LoRA %s:%s", name, weights[0])

    with Timer("Repatch model"):
        model.patch_model()
    s['applied_loras'] = sorted(lora_specs)
    s['model'] = model


NODE_CLASS_MAPPINGS = {
    "EditableCLIPEncode": EditableCLIPEncode,
    "LoRAScheduler": LoRAScheduler,
}
