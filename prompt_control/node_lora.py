import logging
import torch
import inspect

from .utils import unpatch_model, clone_model, set_callback, apply_loras_from_spec
from .parser import parse_prompt_schedules
from .hijack import do_hijack
from comfy.samplers import CFGGuider

log = logging.getLogger("comfyui-prompt-control")

# Use globals to store a cached model to speed up situations where the same LoRA is applied
CACHED_MODEL = None
CACHED_CLONE = None
# This leaks memory, but we'll see if it is a problem...
LORA_CACHE = None

def apply_lora_for_step(model, schedules, step, total_steps, lora_cache):
    # zero-indexed steps, 0 = first step, but schedules are 1-indexed
    sched = schedules.at_step(step + 1, total_steps)
    lora_spec = sched[1]["loras"]
    # mutable dict
    applied_loras = get_state(model, "applied_loras")

    if applied_loras != lora_spec:
        log.debug("At step %s, applying lora_spec %s", step, lora_spec)
        m, _ = apply_loras_from_spec(
            lora_spec,
            model=model,
            orig_model=model,
            cache=lora_cache,
            patch=True,
            applied_loras=applied_loras,
        )
        for k in m.backup.keys():
            if k not in model.backup:
                model.backup[k] = m.backup[k]
        del m
        set_state(model, "applied_loras", lora_spec)


def schedule_lora_common(model, schedules, lora_cache=None):
    do_hijack()
    model.model_options["pc_schedules"] = schedules

    if lora_cache is None:
        lora_cache = {}

    def sampler_cb(orig_sampler, *args, **kwargs):
        split_sampling = args[0].model_options.get("pc_split_sampling")
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
        orig_cb = kwargs["callback"]

        def step_callback(*args, **kwargs):
            current_step = args[0] + start_step
            apply_lora_for_step(model, schedules, current_step, steps, lora_cache)
            if orig_cb:
                return orig_cb(*args, **kwargs)

        kwargs["callback"] = step_callback

        apply_lora_for_step(model, schedules, start_step, steps, lora_cache)

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
            s = orig_sampler(*args, **kwargs)

        unpatch_model(model)

        return s

    set_callback(model, sampler_cb)

    return model


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


# Lambda avoids deepcopy
def get_state(model, key):
    if "pc_model_state" in model.model_options:
        s = model.model_options["pc_model_state"]()
        if key is None:
            return s
        else:
            return s.get(key)
    # Init missing state and retry
    x = dict(applied_loras={})
    model.model_options["pc_model_state"] = lambda: x
    return get_state(model, key)


def set_state(model, key, value):
    s = get_state(model, None)
    s[key] = value


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
        model = self.guider.model_patcher
        backup = get_state(model, "backup")
        if backup is not None:
            model.backup = backup

        def step_callback(*args, **kwargs):
            apply_lora_for_step(
                model,
                self.schedules,
                args[0],
                len(sigmas),
                self.lora_cache,
            )
            if orig_cb:
                return orig_cb(*args, **kwargs)

        kwargs["callback"] = step_callback
        apply_lora_for_step(model, self.schedules, 0, len(sigmas), self.lora_cache)
        try:
            r = self.guider.sample(*args, **kwargs)
        finally:
            # Hold on to the current model state in case the next gen
            # doesn't need to apply more LoRAs
            log.info("Preventing ComfyUI model unpatch")
            set_state(model, "backup", model.backup)
            model.backup = {}
            pass
        return r

def get_cached(model):
    global CACHED_MODEL
    global CACHED_CLONE
    global LORA_CACHE
    if model != CACHED_MODEL:
        LORA_CACHE = {}
        CACHED_CLONE = clone_model(model)
        CACHED_MODEL = model
    return CACHED_CLONE

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
        return (schedule_lora_common(get_cached(model), prompt_schedule, lora_cache=LORA_CACHE),)


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
        return (schedule_lora_common(get_cached(model), schedules, lora_cache=LORA_CACHE),)
