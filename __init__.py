import sys
import os
import lark
from nodes import common_ksampler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.samplers

prompt_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")

def flatten(x):
    if type(x) == str:
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
                before, after, _, when = args
                yield before or () if step <= when else after

            def start(self, args):
                r = ''.join(flatten(args))
                return r

            def plain(self, args):
                yield args[0].value

            def __default__(self, data, children, meta):
                for child in children:
                    yield child

        return AtStep().transform(tree)

    parsed = [[t, at_step(t, tree)] for t in collect(steps, tree)]
    return parsed


def encode_prompts(clip, prompts):
    prev_prompt = None
    res = []
    cache = {}
    for steps, p in prompts:
        # Remove duplicates
        if p != prev_prompt:
            if p not in cache:
                cache[p] = [[clip.encode(p), {}]]
            res.append([steps, cache[p]])
        prev_prompt = p
    return res


class KSamplerPromptEditing:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive_prompt": ("STRING", {"multiline": True}),
                    "negative_prompt": ("STRING", {"multiline": True}),
                    "latent_image": ("LATENT", ),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "clip": ("CLIP", ),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Local/nodes"
    def sample(self, clip,model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive_prompt, negative_prompt, latent_image, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        
        prompt_schedules = parse_prompt_schedules(positive_prompt,steps)
        encoded = encode_prompts(clip, prompt_schedules)
        print(f"Expanded prompt schedules: {prompt_schedules}")
        negative = [[clip.encode(negative_prompt), {}]]
        current_step = 0
        out = latent_image
        full_denoise = False
        disable_noise = disable_noise
        for until, cond in encoded:
            if current_step >= steps:
                break
            if until >= steps:
                full_denoise = force_full_denoise
            print(f"Sampling for {until - current_step} steps, {full_denoise=}, {disable_noise=}")
            out, = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, cond, negative, out, denoise=denoise, disable_noise=disable_noise, start_step=current_step, last_step=min(until, steps), force_full_denoise=full_denoise)
            current_step = until
            # Don't add noise multiple times
            disable_noise = True

        return (out,)
        

NODE_CLASS_MAPPINGS = {
    "KSamplerPromptEditing":KSamplerPromptEditing,
}
