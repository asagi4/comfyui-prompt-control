import logging

import comfy.hooks
import comfy.utils
import folder_paths
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict

from .attention_couple_ppm import AttentionCoupleHook
from .parser import parse_prompt_schedules
from .utils import consolidate_schedule

log = logging.getLogger("comfyui-prompt-control")


class PCLoraHooksFromText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"text": ("STRING",)},
        }

    RETURN_TYPES = ("HOOKS",)
    OUTPUT_TOOLTIPS = ("set of hooks created from the prompt schedule",)
    CATEGORY = "promptcontrol/v2"
    FUNCTION = "apply"
    EXPERIMENTAL = True

    def apply(self, text):
        prompt_schedule = parse_prompt_schedules(text)
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


class PCAttentionCoupleBatchNegative(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "positive": (IO.CONDITIONING, {}),
                "negative": (IO.CONDITIONING, {}),
            },
        }

    RETURN_TYPES = (IO.CONDITIONING, IO.CONDITIONING)
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "promptcontrol/v2"
    FUNCTION = "batch"
    EXPERIMENTAL = True

    # May cause side-effects?
    # TODO: Support scheduling in negative prompt
    def batch(self, positive, negative):
        if len(negative) != 1:
            log.warning("Batching scheduled negatives is not supported yet")
            return (positive, negative)

        negative_batch = []
        for p in positive:
            n = [negative[0][0], negative[0][1].copy()]
            n_hook_group: comfy.hooks.HookGroup = n[1].get("hooks", comfy.hooks.HookGroup()).clone()
            p_hook_group: comfy.hooks.HookGroup = p[1].get("hooks", comfy.hooks.HookGroup())
            attn_couple = [hook for hook in p_hook_group.hooks if isinstance(hook, AttentionCoupleHook)]
            for hook in attn_couple:
                n_hook_group.add(hook)
            n[1]["hooks"] = p_hook_group if n_hook_group.hooks == p_hook_group.hooks else n_hook_group
            n[1]["start_percent"] = p[1].get("start_percent", 0.0)
            n[1]["end_percent"] = p[1].get("end_percent", 1.0)
            negative_batch.append(n)

        return (positive, negative_batch)


NODE_CLASS_MAPPINGS = {
    "PCLoraHooksFromText": PCLoraHooksFromText,
    "PCAttentionCoupleBatchNegative": PCAttentionCoupleBatchNegative,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PCLoraHooksFromText": "PC: LoRA Hooks From Text (non-lazy)",
    "PCAttentionCoupleBatchNegative": "PC: Attention Couple (batch negative)",
}
