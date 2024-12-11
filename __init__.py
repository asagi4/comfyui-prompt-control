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

from .prompt_control.nodes_lazy import NODE_CLASS_MAPPINGS as lazy_mappings, NODE_DISPLAY_NAME_MAPPINGS as lazy_display

NODE_CLASS_MAPPINGS.update(lazy_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(lazy_display)

import importlib

if importlib.util.find_spec("comfy.hooks"):
    from .prompt_control.node_hooks import (
        NODE_CLASS_MAPPINGS as hook_mappings,
        NODE_DISPLAY_NAME_MAPPINGS as hook_display,
    )

    NODE_CLASS_MAPPINGS.update(hook_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(hook_display)
else:
    log.warning(
        "Your ComfyUI version is too old, can't import comfy.hooks for PCEncodeSchedule and PCLoraHooksFromSchedule. Update your installation."
    )

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


NODE_CLASS_MAPPINGS.update(
    {
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
)
