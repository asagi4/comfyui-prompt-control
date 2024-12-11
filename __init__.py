import os
import sys
import logging

from .prompt_control.legacy.node_clip import EditableCLIPEncode, ScheduleToCond
from .prompt_control.legacy.node_lora import LoRAScheduler, ScheduleToModel, PCSplitSampling, PCWrapGuider
from .prompt_control.legacy.node_other import (
    PromptToSchedule,
    FilterSchedule,
    PCScheduleSettings,
    PCScheduleAddMasks,
    PCApplySettings,
    PCPromptFromSchedule,
)
from .prompt_control.legacy.node_aio import PromptControlSimple

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

log = logging.getLogger("comfyui-prompt-control-legacy")
log.propagate = False
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(levelname)s] PromptControl (LEGACY VERSION): %(message)s"))
    log.addHandler(h)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


NODE_CLASS_MAPPINGS.update(
    {
        "PromptControlSimple": PromptControlSimple,
        "PromptToSchedule": PromptToSchedule,
        "PCSplitSampling": PCSplitSampling,
        "PCPromptFromSchedule": PCPromptFromSchedule,
        "PCWrapGuider": PCWrapGuider,
        "FilterSchedule": FilterSchedule,
        "ScheduleToCond": ScheduleToCond,
        "ScheduleToModel": ScheduleToModel,
        "EditableCLIPEncode": EditableCLIPEncode,
        "LoRAScheduler": LoRAScheduler,
    }
)
