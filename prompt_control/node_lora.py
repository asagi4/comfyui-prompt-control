from . import utils as utils
import comfy.sd
from .utils import untuple
from .parser import parse_prompt_schedules
from .hijack import do_hijack, get_aitemplate_module

import logging

log = logging.getLogger("comfyui-prompt-control")


# AITemplate support
def set_callback(model, cb):
    model = untuple(model)
    setattr(model, "prompt_control_callback", cb)


def get_lora_keymap(model=None, clip=None):
    key_map = {}
    if model:
        model = untuple(model)
        key_map = comfy.sd.model_lora_keys_unet(model.model)
    if clip:
        key_map = comfy.sd.model_lora_keys_clip(clip.cond_stage_model, key_map)
    return key_map


def unpatch_model(model):
    untuple(model).unpatch_model()


def clone_model(model):
    if isinstance(model, tuple):
        return (model[0].clone(), model[1])
    else:
        return model.clone()


def add_patches(model, patches, weight):
    untuple(model).add_patches(patches, weight)


def patch_model(model):
    if isinstance(model, tuple):
        m = model[0]
        m.patch_model()
        mod = get_aitemplate_module()
        l = mod.AITemplate.loader
        if hasattr(l, "pc_applied_module"):
            log.info("Applying AITemplate unet")
            l.apply_unet(
                aitemplate_module=l.pc_applied_module,
                unet=l.compvis_unet(m.model.state_dict()),
                in_channels=m.model.diffusion_model.in_channels,
                conv_in_key="conv_in_weight",
            )
    else:
        model.patch_model(model.load_device)
        model.model.to(model.load_device)


class NoOut(object):
    def write(*args):
        pass

    def flush(*args):
        pass


def load_lora(model, lora, weight, key_map, clone=True):
    # Hack to temporarily override printing to stdout to stop log spam
    def noop(*args):
        pass

    p = print
    __builtins__["print"] = noop
    loaded = comfy.sd.load_lora(lora, key_map)
    __builtins__["print"] = p
    if clone:
        model = clone_model(model)
    add_patches(model, loaded, weight)
    return model


def apply_loras_to_model(model, orig_model, lora_specs, loaded_loras, patch=True):
    keymap = get_lora_keymap(model=model)
    if patch:
        unpatch_model(model)
        model = clone_model(orig_model)

    for name, params in lora_specs.items():
        if name not in loaded_loras:
            continue
        model = load_lora(model, loaded_loras[name], params['weight'], keymap)
        log.info("Loaded LoRA %s:%s", name, params['weight'])

    if patch:
        patch_model(model)

    return model


class LoRAScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "cutoff": ("FLOAT", {"min": 0.00, "max": 1.00, "default": 0.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, model, text, cutoff=0.0):
        do_hijack()
        orig_model = clone_model(model)
        schedules = parse_prompt_schedules(text)
        schedules = [(t, s) for t, s in schedules if t >= cutoff]
        loaded_loras = {}
        loaded_loras = utils.load_loras_from_schedule(schedules, loaded_loras)

        def sampler_cb(orig_sampler, *args, **kwargs):
            state = {}
            steps = args[2]
            start_step = kwargs["start_step"] or 0
            # The model patcher may change if LoRAs are applied
            state["model"] = args[0]
            state["applied_loras"] = []

            orig_cb = kwargs["callback"]

            def apply_lora_for_step(step, patch=True):
                # zero-indexed steps, 0 = first step, but schedules are 1-indexed
                sched = utils.schedule_for_step(steps, step + 1, schedules)
                lora_spec = sorted(sched[1]["loras"])

                if state["applied_loras"] != lora_spec:
                    log.debug("At step %s, applying lora_spec %s", step, lora_spec)
                    state["model"] = apply_loras_to_model(state["model"], orig_model, lora_spec, loaded_loras, patch)
                    state["applied_loras"] = lora_spec

            def step_callback(*args, **kwargs):
                current_step = args[0] + start_step
                # Callbacks are called *after* the step so apply for next step
                apply_lora_for_step(current_step + 1)
                if orig_cb:
                    return orig_cb(*args, **kwargs)

            kwargs["callback"] = step_callback

            # First step of sampler applies patch
            apply_lora_for_step(start_step, patch=False)
            args = list(args)
            args[0] = state["model"]
            s = orig_sampler(*args, **kwargs)

            if state["applied_loras"]:
                log.info("Sampling done with leftover LoRAs, unpatching")
                # state may have been modified
                unpatch_model(state["model"])

            return s

        set_callback(orig_model, sampler_cb)

        return (orig_model,)
