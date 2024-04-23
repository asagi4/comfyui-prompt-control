import logging
import torch
import inspect

from .utils import unpatch_model, clone_model, set_callback, apply_loras_from_spec
from .parser import parse_prompt_schedules
from .hijack import do_hijack
from comfy.samplers import CFGGuider

log = logging.getLogger("comfyui-prompt-control")


def apply_lora_for_step(schedules, step, total_steps, state, original_model, lora_cache, patch=True):
    # zero-indexed steps, 0 = first step, but schedules are 1-indexed
    sched = schedules.at_step(step + 1, total_steps)
    lora_spec = sched[1]["loras"]

    if state["applied_loras"] != lora_spec:
        log.debug("At step %s, applying lora_spec %s", step, lora_spec)
        m, _ = apply_loras_from_spec(
            lora_spec,
            model=state["model"],
            orig_model=original_model,
            cache=lora_cache,
            patch=patch,
            applied_loras=state["applied_loras"],
        )
        state["model"] = m
        state["applied_loras"] = lora_spec


def schedule_lora_common(orig_model, schedules, lora_cache=None):
    do_hijack()
    orig_model.model_options["pc_schedules"] = schedules

    if lora_cache is None:
        lora_cache = {}

    def sampler_cb(orig_sampler, *args, **kwargs):
        split_sampling = args[0].model_options.get("pc_split_sampling")
        state = {}
        # For custom samplers, sigmas is not a keyword argument. Do the check this way to fall back to old behaviour if other hijacks exist.
        if "sigmas" in inspect.getfullargspec(orig_sampler).args:
            steps = len(args[4])
            log.info(
                "SamplerCustom detected, number of steps not available. LoRA schedules will be calculated based on the number of sigmas (%s)",
                steps,
            )
        else:
            steps = args[2]
        start_step = kwargs.get("start_step") or 0
        # The model patcher may change if LoRAs are applied
        state["model"] = args[0]
        state["applied_loras"] = {}

        orig_cb = kwargs["callback"]

        def step_callback(*args, **kwargs):
            current_step = args[0] + start_step
            apply_lora_for_step(schedules, current_step, steps, state, orig_model, lora_cache, patch=True)
            if orig_cb:
                return orig_cb(*args, **kwargs)

        kwargs["callback"] = step_callback

        apply_lora_for_step(schedules, start_step, steps, state, orig_model, lora_cache, patch=True)

        def filter_conds(conds, t, start_t, end_t):
            r = []
            for c in conds:
                x = c[1].copy()
                start_at = round(x["start_percent"], 2)
                end_at = round(x["end_percent"], 2)
                # Take any cond that has any effect before end_t, since the percentages may not perfectly match
                if end_t > start_at and end_t <= end_at:
                    del x["start_percent"]
                    del x["end_percent"]
                    r.append([c[0].clone(), x])
                else:
                    log.debug("Rejecting cond (%s, %s) between (%s, %s)", start_at, end_at, start_t, end_t)
            if len(r) == 0:
                log.error("No %s conds between (%s, %s); Try adjusting your steps", t, start_t, end_t)
            return r

        def get_steps(conds):
            for c in conds:
                yield round(c[1].get("end_percent", 0), 2)

        if split_sampling:
            actual_end_step = kwargs["last_step"] or steps
            first_step = True
            s = args[8]
            all_steps = sorted(set(int(steps * i) for i in [1.0] + list(get_steps(args[6])) + list(get_steps(args[7]))))
            for end_step in all_steps:
                if end_step <= start_step:
                    continue
                start_t = round(start_step / steps, 2)
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

        unpatch_model(state["model"])

        return s

    set_callback(orig_model, sampler_cb)

    return orig_model


class PCWrapGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "guider": ("GUIDER",),
            },
        }

    CATEGORY = "promptcontrol"
    FUNCTION = "apply"
    RETURN_TYPES = ("GUIDER",)

    def apply(self, guider):
        return (PCGuider(guider),)


class PCGuider(CFGGuider):
    def __init__(self, original_guider):
        if "pc_schedules" not in original_guider.model_patcher.model_options:
            raise ValueError(
                "The guider passed to PCWrapGuider must contain a Model that has schedules applied. Use ScheduleToModel"
            )
        self.schedules = original_guider.model_patcher.model_options["pc_schedules"]
        self.guider = original_guider
        self.lora_cache = {}
        # sets self.model_patcher
        super().__init__(original_guider.model_patcher)

    def sample(self, *args, **kwargs):
        orig_cb = kwargs["callback"]
        sigmas = args[3]
        state = {"model": self.guider.model_patcher, "applied_loras": {}}

        def step_callback(*args, **kwargs):
            apply_lora_for_step(
                self.schedules,
                args[0],
                len(sigmas),
                state,
                self.guider.model_patcher,
                self.lora_cache,
                patch=True,
            )
            if orig_cb:
                return orig_cb(*args, **kwargs)

        kwargs["callback"] = step_callback
        apply_lora_for_step(
            self.schedules, 0, len(sigmas), state, self.guider.model_patcher, self.lora_cache, patch=True
        )
        try:
            r = self.guider.sample(*args, **kwargs)
        finally:
            unpatch_model(state["model"])
        return r


class ScheduleToModel:
    cached_model = None
    cached_clone = None
    lora_cache = None

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
        if model != self.cached_model:
            self.cached_clone = clone_model(model)
            self.cached_model = model
            self.lora_cache = {}
        return (schedule_lora_common(self.cached_clone, prompt_schedule, lora_cache=self.lora_cache),)


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
        model = clone_model(model)
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
        model = model.clone()
        return (schedule_lora_common(model, schedules),)
