from .utils import unpatch_model, clone_model, set_callback, apply_loras_to_model
from .parser import parse_prompt_schedules
from .hijack import do_hijack

import logging

log = logging.getLogger("comfyui-prompt-control")


def schedule_lora_common(model, schedules, lora_cache=None):
    do_hijack()
    orig_model = clone_model(model)
    loaded_loras = schedules.load_loras(lora_cache)

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
            sched = schedules.at_step(steps, step + 1)
            lora_spec = sched[1]["loras"]

            if state["applied_loras"] != lora_spec:
                log.debug("At step %s, applying lora_spec %s", step, lora_spec)
                state["model"] = apply_loras_to_model(state["model"], orig_model, lora_spec, loaded_loras, patch)
                state["applied_loras"] = lora_spec

        def step_callback(*args, **kwargs):
            current_step = args[0] + start_step
            apply_lora_for_step(current_step)
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

    return orig_model


class ScheduleToModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt_schedule": ("PROMPT_SCHEDULE",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, model, prompt_schedule):
        return (schedule_lora_common(model, prompt_schedule),)


class LoRAScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "promptcontrol/old"
    FUNCTION = "apply"

    def apply(self, model, text):
        schedules = parse_prompt_schedules(text)
        return (schedule_lora_common(model, schedules),)
