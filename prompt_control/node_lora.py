from .utils import unpatch_model, clone_model, set_callback, apply_loras_to_model
from .parser import parse_prompt_schedules
from .hijack import do_hijack

import torch
import logging

log = logging.getLogger("comfyui-prompt-control")


def schedule_lora_common(model, schedules, lora_cache=None):
    do_hijack()
    orig_model = clone_model(model)
    loaded_loras = schedules.load_loras(lora_cache)

    def sampler_cb(orig_sampler, *args, **kwargs):
        split_sampling = args[0].model_options.get("pc_split_sampling")
        state = {}
        steps = args[2]
        start_step = kwargs["start_step"] or 0
        # The model patcher may change if LoRAs are applied
        state["model"] = args[0]
        state["applied_loras"] = []

        orig_cb = kwargs["callback"]

        def apply_lora_for_step(step, patch=True):
            # zero-indexed steps, 0 = first step, but schedules are 1-indexed
            sched = schedules.at_step(step + 1, steps)
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

        def filter_conds(conds, t, start_t, end_t):
            r = []
            for c in conds:
                x = c[1].copy()
                start_at = round(1 - x["start_percent"], 2)
                end_at = round(1 - x["end_percent"], 2)
                if start_t >= start_at and end_t <= end_at:
                    del x["start_percent"]
                    del x["end_percent"]
                    r.append([c[0].clone(), x])
            if len(r) == 0:
                log.error("No %s conds between (%s, %s); Try adjusting your steps", t, start_t, end_t)
            return r

        if split_sampling:
            actual_end_step = kwargs["last_step"] or steps
            first_step = True
            s = args[8]
            while start_step < actual_end_step:
                end_t, _ = schedules.at_step(start_step + 1, total_steps=steps)
                start_t = round(start_step / steps, 2)
                if end_t == start_t:
                    end_t, _ = schedules.at_step(start_step + 1, total_steps=steps)
                end_step = int(steps * end_t)
                end_t = round(end_step / steps, 2)
                new_kwargs = kwargs.copy()
                new_args = list(args)
                new_args[0] = state["model"]
                new_args[6] = filter_conds(new_args[6], "positive", start_t, end_t)
                new_args[7] = filter_conds(new_args[7], "negative", start_t, end_t)
                new_args[8] = s
                log.info("Sampling from %s to %s (total: %s)", start_step, end_step, actual_end_step)
                new_kwargs["start_step"] = start_step
                new_kwargs["last_step"] = end_step
                if end_step >= min(steps, actual_end_step):
                    new_kwargs["force_full_denoise"] = kwargs["force_full_denoise"]
                else:
                    new_kwargs["force_full_denoise"] = False

                if not first_step:
                    # disable_noise apparently does nothing currently, we need to override noise in args
                    new_kwargs["disable_noise"] = True
                    new_args[1] = torch.zeros_like(s)

                s = orig_sampler(*new_args, **new_kwargs)
                start_step = end_step
                first_step = False
        else:
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
            },
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, model, prompt_schedule):
        return (schedule_lora_common(model, prompt_schedule),)


class PCSplitSampling:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "split_sampling": (["enable", "disable"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, model, split_sampling):
        model = model.clone()
        model.model_options["pc_split_sampling"] = split_sampling == "enable"
        return (model,)


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
