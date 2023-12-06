import logging
from .parser import parse_prompt_schedules

log = logging.getLogger("comfyui-prompt-control")


class FilterSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"prompt_schedule": ("PROMPT_SCHEDULE",)},
            "optional": {
                "tags": ("STRING", {"default": ""}),
                "start": ("FLOAT", {"min": 0.00, "max": 1.00, "default": 0.0, "step": 0.01}),
                "end": ("FLOAT", {"min": 0.00, "max": 1.00, "default": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("PROMPT_SCHEDULE",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, prompt_schedule, tags="", start=0.0, end=1.0):
        p = prompt_schedule.with_filters(tags, start=start, end=end)
        log.debug(
            f"Filtered {prompt_schedule.parsed_prompt} with: ({tags}, {start}, {end}); the result is %s",
            p.parsed_prompt,
        )
        return (p,)


class PromptToSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("PROMPT_SCHEDULE",)
    CATEGORY = "promptcontrol"
    FUNCTION = "parse"

    def parse(self, text, filter_tags=""):
        schedules = parse_prompt_schedules(text)
        return (schedules,)
