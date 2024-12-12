import logging
from .prompts import encode_prompt
from .parser import parse_prompt_schedules

log = logging.getLogger("comfyui-prompt-control")


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


class PCEncodeSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP",), "prompt_schedule": ("PC_SCHEDULE",)},
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol/schedule"
    FUNCTION = "apply"

    def apply(self, clip, prompt_schedule):
        return (encode_schedule(clip, prompt_schedule),)


class FilterSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"prompt_schedule": ("PC_SCHEDULE",)},
            "optional": {
                "tags": ("STRING", {"default": ""}),
                "start": ("FLOAT", {"min": 0.00, "max": 1.00, "default": 0.0, "step": 0.01}),
                "end": ("FLOAT", {"min": 0.00, "max": 1.00, "default": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("PC_SCHEDULE",)
    CATEGORY = "promptcontrol/schedule"
    FUNCTION = "apply"

    def apply(self, prompt_schedule, tags="", start=0.0, end=1.0):
        p = prompt_schedule.with_filters(tags, start=start, end=end)
        log.debug(
            f"Filtered {prompt_schedule.parsed_prompt} with: ({tags}, {start}, {end}); the result is %s",
            p.parsed_prompt,
        )
        return (p,)


class PCApplySettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"prompt_schedule": ("PC_SCHEDULE",), "settings": ("PC_SETTINGS",)}}

    RETURN_TYPES = ("PC_SCHEDULE",)
    CATEGORY = "promptcontrol/schedule"
    FUNCTION = "apply"

    def apply(self, prompt_schedule, settings):
        return (prompt_schedule.with_filters(defaults=settings),)


class PCScheduleAddMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"prompt_schedule": ("PC_SCHEDULE",)},
            "optional": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "mask3": ("MASK",),
                "mask4": ("MASK",),
            },
        }

    RETURN_TYPES = ("PC_SCHEDULE",)
    CATEGORY = "promptcontrol/schedule"
    FUNCTION = "apply"

    def apply(self, prompt_schedule, mask1=None, mask2=None, mask3=None, mask4=None):
        p = prompt_schedule.clone()
        p.add_masks(mask1, mask2, mask3, mask4)
        return (p,)


class PCSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "mask_width": ("INT", {"default": 512, "min": 64, "max": 4096 * 4}),
                "mask_height": ("INT", {"default": 512, "min": 64, "max": 4096 * 4}),
                "sdxl_width": ("INT", {"default": 1024, "min": 0, "max": 4096 * 4}),
                "sdxl_height": ("INT", {"default": 1024, "min": 0, "max": 4096 * 4}),
                "sdxl_target_w": ("INT", {"default": 1024, "min": 0, "max": 4096 * 4}),
                "sdxl_target_h": ("INT", {"default": 1024, "min": 0, "max": 4096 * 4}),
                "sdxl_crop_w": ("INT", {"default": 0, "min": 0, "max": 4096 * 4}),
                "sdxl_crop_h": ("INT", {"default": 0, "min": 0, "max": 4096 * 4}),
            },
        }

    RETURN_TYPES = ("PC_SETTINGS",)
    CATEGORY = "promptcontrol/schedule"
    FUNCTION = "apply"

    def apply(
        self,
        steps=0,
        mask_width=512,
        mask_height=512,
        sdxl_width=1024,
        sdxl_height=1024,
        sdxl_target_w=1024,
        sdxl_target_h=1024,
        sdxl_crop_w=0,
        sdxl_crop_h=0,
    ):
        settings = {
            "steps": steps,
            "mask_width": mask_width,
            "mask_height": mask_height,
            "sdxl_width": sdxl_width,
            "sdxl_height": sdxl_height,
            "sdxl_twidth": sdxl_target_w,
            "sdxl_theight": sdxl_target_h,
            "sdxl_cwidth": sdxl_crop_w,
            "sdxl_cheight": sdxl_crop_h,
        }
        return (settings,)


class PCPromptFromSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_schedule": ("PC_SCHEDULE",),
                "at": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"tags": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "promptcontrol/schedule"
    FUNCTION = "apply"

    def apply(self, prompt_schedule, at, tags=""):
        p = prompt_schedule.with_filters(tags, start=at, end=at).parsed_prompt[-1][1]
        log.info("Prompt at %s:\n%s", at, p["prompt"])
        log.info("LoRAs: %s", p["loras"])
        return (p["prompt"],)


class PromptToSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("PC_SCHEDULE",)
    CATEGORY = "promptcontrol/schedule"
    FUNCTION = "parse"

    def parse(self, text, settings=None):
        schedules = parse_prompt_schedules(text)
        return (schedules,)


NODE_CLASS_MAPPINGS = {
    "PCPromptToSchedule": PromptToSchedule,
    "PCSettings": PCScheduleSettings,
    "PCScheduleAddMasks": PCScheduleAddMasks,
    "PCApplySettings": PCApplySettings,
    "PCPromptFromSchedule": PCPromptFromSchedule,
    "PCFilterSchedule": FilterSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {}