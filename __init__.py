import sys
import os
import lark
from nodes import common_ksampler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.samplers

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

# Mostly lifted from A1111
def parse_prompt_schedules(prompt, steps):
    try:
        tree = prompt_parser.parse(prompt)
    except lark.exceptions.LarkError as e:
        print("Prompt editing parse error:", e)
        return [[steps, prompt]]

    def collect(steps, tree):
        res = [steps]
        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                # Last element in []
                tree.children[-1] = float(tree.children[-1])
                if tree.children[-1] < 1:
                    tree.children[-1] = round(tree.children[-1]*steps)
                res.append(int(tree.children[-1]))

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
                    else:
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

    parsed = [[t, at_step(t, tree)] for t in collect(steps, tree)]
    return parsed


def encode_prompts(clip, schedules):
    cache = {}
    for step, s in schedules:
        p = s["prompt"]
        if p not in cache:
            cache[p] = [[clip.encode(p), {}]]
        s["cond"] = cache[p]
    return schedules


class EditablePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"clip": ("CLIP",),
                 "positive": ("STRING", {"multiline": True}),
                 "negative": ("STRING", {"multiline": True}),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                }
            }
    RETURN_TYPES = ("COND_SCHEDULE",)
    CATEGORY= "nodes"
    FUNCTION = "parse"

    def parse(self, clip, positive, negative, steps):
        parsed = parse_prompt_schedules(positive, steps)
        positive = encode_prompts(clip, parsed)
        parsed = parse_prompt_schedules(negative, steps)
        negative = encode_prompts(clip, parsed)

        return ({"steps": steps, "positive": positive, "negative": negative},)

class KSamplerCondSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
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
    def sample(self, model, add_noise, noise_seed, cfg, sampler_name, scheduler, cond_schedule, latent_image, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        steps = cond_schedule["steps"]
        positive = cond_schedule["positive"]
        negative = cond_schedule["negative"]
        
        current_step = 0
        out = latent_image
        full_denoise = False
        disable_noise = disable_noise

        def for_step(step, schedules):
            for end, s in schedules:
                if end > step:
                    return [end, s]
            return schedules[-1]

        while current_step < steps:
            nsteps, neg = for_step(current_step, negative)
            psteps, pos = for_step(current_step, positive)
            until = min(nsteps, psteps)
            if until >= steps:
                full_denoise = force_full_denoise
            print(f"Sampling with {pos['prompt']=} and {neg['prompt']=} for {until - current_step} steps, {full_denoise=}, {disable_noise=}")
            out, = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, pos["cond"], neg["cond"], out, denoise=denoise, disable_noise=disable_noise, start_step=current_step, last_step=min(until, steps), force_full_denoise=full_denoise)
            current_step = until
            # Don't add noise multiple times
            disable_noise = True

        return (out,)


NODE_CLASS_MAPPINGS = {
    "KSamplerCondSchedule": KSamplerCondSchedule,
    "EditablePrompt": EditablePrompt,
}
