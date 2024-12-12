import logging
import comfy.utils
import comfy.hooks
import folder_paths
from .prompts import encode_prompt
from .utils import consolidate_schedule

log = logging.getLogger("comfyui-prompt-control")


class PCLoraHooksFromSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"prompt_schedule": ("PC_SCHEDULE",)},
        }

    RETURN_TYPES = ("HOOKS",)
    OUTPUT_TOOLTIPS = ("set of hooks created from the prompt schedule",)
    CATEGORY = "promptcontrol/schedule"
    FUNCTION = "apply"

    def apply(self, prompt_schedule):
        consolidated = consolidate_schedule(prompt_schedule)
        hooks = lora_hooks_from_schedule(consolidated, {})
        return (hooks,)


def lora_hooks_from_schedule(schedules, non_scheduled):
    start_pct = 0.0
    lora_cache = {}
    all_hooks = []

    def create_hook(loraspec, start_pct, end_pct, non_scheduled):
        nonlocal lora_cache
        hooks = []
        hook_kf = comfy.hooks.HookKeyframeGroup()
        for path, info in loras.items():
            if non_scheduled.get(path) == info:
                log.info("Skipping %s from hook, it's loaded directly on model", path)
                continue
            if path not in lora_cache:
                lora_cache[path] = comfy.utils.load_torch_file(
                    folder_paths.get_full_path("loras", path), safe_load=True
                )
            new_hook = comfy.hooks.create_hook_lora(
                lora_cache[path], strength_model=info["weight"], strength_clip=info["weight_clip"]
            )
            # Set hook_ref so that identical hooks compare equal
            new_hook.hooks[0].hook_ref = f"pc-{path}-{info['weight']}-{info['weight_clip']}"
            hooks.append(new_hook)
        if start_pct > 0.0:
            kf = comfy.hooks.HookKeyframe(strength=0.0, start_percent=0.0)
            hook_kf.add(kf)
        kf = comfy.hooks.HookKeyframe(strength=1.0, start_percent=start_pct)
        hook_kf.add(kf)
        if end_pct < 1.0:
            kf = comfy.hooks.HookKeyframe(strength=0.0, start_percent=end_pct)
            hook_kf.add(kf)
        hooks = comfy.hooks.HookGroup.combine_all_hooks(hooks)
        if hooks:
            hooks.set_keyframes_on_hooks(hook_kf=hook_kf)
        return hooks

    for end_pct, loras in schedules:
        log.info("Creating LoRA hook from %s to %s: %s", start_pct, end_pct, loras)
        hook = create_hook(loras, start_pct, end_pct, non_scheduled)
        all_hooks.append(hook)
        start_pct = end_pct

    del lora_cache

    all_hooks = [x for x in all_hooks if x]

    if all_hooks:
        hooks = comfy.hooks.HookGroup.combine_all_hooks(all_hooks)
        return hooks


def encode_schedule(clip, schedules):
    start_pct = 0.0
    conds = []
    for end_pct, c in schedules:
        if start_pct < end_pct:
            prompt = c["prompt"]
            cond = encode_prompt(clip, prompt, start_pct, end_pct, schedules.defaults, schedules.masks)
            conds.extend(cond)
        start_pct = end_pct

    return conds


NODE_CLASS_MAPPINGS = {
    "PCLoraHooksFromSchedule": PCLoraHooksFromSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PCLoraHooksFromSchedule": "PC Create LoRA Hooks from Schedule",
}
