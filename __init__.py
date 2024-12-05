import os
import sys
import logging

from .prompt_control.node_clip import EditableCLIPEncode, ScheduleToCond
from .prompt_control.node_lora import LoRAScheduler, ScheduleToModel, PCSplitSampling, PCWrapGuider
from .prompt_control.node_other import (
    PromptToSchedule,
    FilterSchedule,
    PCScheduleSettings,
    PCScheduleAddMasks,
    PCApplySettings,
    PCPromptFromSchedule,
)
from .prompt_control.node_aio import PromptControlSimple


log = logging.getLogger("comfyui-prompt-control")
log.propagate = False
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(levelname)s] PromptControl: %(message)s"))
    log.addHandler(h)

if os.environ.get("COMFYUI_PC_DEBUG"):
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)

import importlib

if importlib.util.find_spec("comfy.hooks"):
    from .prompt_control.node_hooks import PCLoraHooksFromSchedule, PCEncodeSchedule, PCEncodeSingle

    maps = {
        "PCLoraHooksFromSchedule": PCLoraHooksFromSchedule,
        "PCEncodeSchedule": PCEncodeSchedule,
        "PCEncodeSingle": PCEncodeSingle,
    }
else:
    log.warning(
        "Your ComfyUI version is too old, can't import comfy.hooks for PCEncodeSchedule and PCLoraHooksFromSchedule. Update your installation."
    )
    maps = {}

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


NODE_CLASS_MAPPINGS = {
    "PromptControlSimple": PromptControlSimple,
    "PromptToSchedule": PromptToSchedule,
    "PCSplitSampling": PCSplitSampling,
    "PCScheduleSettings": PCScheduleSettings,
    "PCScheduleAddMasks": PCScheduleAddMasks,
    "PCApplySettings": PCApplySettings,
    "PCPromptFromSchedule": PCPromptFromSchedule,
    "PCWrapGuider": PCWrapGuider,
    "FilterSchedule": FilterSchedule,
    "ScheduleToCond": ScheduleToCond,
    "ScheduleToModel": ScheduleToModel,
    "EditableCLIPEncode": EditableCLIPEncode,
    "LoRAScheduler": LoRAScheduler,
}

NODE_CLASS_MAPPINGS.update(maps)
