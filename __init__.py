import sys
import os
import lark
from nodes import common_ksampler
from pathlib import Path

import logging

log = logging.getLogger('comfyui-prompt-scheduling')

import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

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


class EditableCLIPEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"clip": ("CLIP",),
                 "text": ("STRING", {"multiline": True}),
                }
               }
    RETURN_TYPES = ('CONDITIONING',)
    CATEGORY = 'nodes'
    FUNCTION = 'parse'

    def parse(self, clip, text):
        parsed = parse_prompt_schedules(text)
        start_pct = 0.0
        conds = []
        for end_pct, c in parsed:
            # TODO: LoRA
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

class KSamplerCondSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "clip": ("CLIP",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "cond_schedule": ("COND_SCHEDULE",),
                    "latent_image": ("LATENT", ),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Local/nodes"

    def __init__(self):
        self.loaded_loras = {}
        self.applied_loras = []
        self.model = None
        self.clip = None
        self.current_model = None
        self.current_clip = None

    def load_loras(self, lora_specs):
        # TODO: unload loras
        filenames = [Path(f) for f in folder_paths.get_filename_list("loras")]
        names = set(name for name, _ in lora_specs)
        for name in names:
            found = False
            for f in filenames:
                if f.stem == name:
                    full_path = folder_paths.get_full_path("loras", str(f))
                    self.loaded_loras[name] = comfy.utils.load_torch_file(full_path, safe_load=True)
                    found = True
                    break
            if not found:
                log.warning("Lora %s not found", name)

    def apply_loras(self, lora_specs):
        need_reload = False
        if len(lora_specs) != len(self.applied_loras):
            need_reload = True
        for spec in lora_specs:
            name, args = spec
            if name not in  self.loaded_loras:
                continue
            # Ensure it has at least 2 items
            args += args[:1]
            unetw, clipw = args[:2]
            # TODO: model changing
            if not (name, unetw, clipw) in self.applied_loras:
                need_reload = True
                break
        if need_reload:
            self.applied_loras = []
            with Timer("unpatch models"):
                self.current_model.unpatch_model()
                self.current_clip.unpatch_model()
            with Timer("clone models"):
                self.current_model = self.model.clone()
                self.current_clip = self.clip.clone()
            for spec in lora_specs:
                name, args = spec
                # Ensure it has at least 2 items
                args += args[:1]
                unetw, clipw = args[:2]
                self.current_model, self.current_clip = comfy.sd.load_lora_for_models(self.current_model, self.current_clip, self.loaded_loras[name], unetw, clipw)
                self.applied_loras += [(name, unetw, clipw)]
            with Timer("patch_models"):
                self.current_model.patch_model()


    def sample(self, model, clip, add_noise, noise_seed, cfg, sampler_name, scheduler, cond_schedule, latent_image, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        steps = cond_schedule["steps"]
        positive = cond_schedule["positive"]
        negative = cond_schedule["negative"]

        lora_specs = []
        for step, sched in positive:
            if sched["loras"]:
                lora_specs.extend(sched["loras"])

        self.load_loras(lora_specs)
        
        
        current_step = 0
        out = latent_image
        full_denoise = False
        disable_noise = disable_noise

        def for_step(step, schedules):
            for end, s in schedules:
                if end > step:
                    return [end, s]
            return schedules[-1]

        self.current_model = model.clone()
        self.model = model
        self.current_clip = clip.clone()
        self.clip = clip
        while current_step < steps:
            nsteps, neg = for_step(current_step, negative)
            psteps, pos = for_step(current_step, positive)
            until = min(nsteps, psteps)
            if until >= steps:
                full_denoise = force_full_denoise

            self.apply_loras(pos["loras"])

            log.info(f"Sampling with {pos['prompt']=} and {neg['prompt']=} for {until - current_step} steps, {full_denoise=}, {disable_noise=}")
            out, = common_ksampler(self.current_model, noise_seed, steps, cfg, sampler_name, scheduler, pos["cond"], neg["cond"], out, denoise=denoise, disable_noise=disable_noise, start_step=current_step, last_step=min(until, steps), force_full_denoise=full_denoise)
            current_step = until
            # Don't add noise multiple times
            disable_noise = True

        return (out,)


NODE_CLASS_MAPPINGS = {
    "EditableCLIPEncode": EditableCLIPEncode,
}
