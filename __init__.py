import os
import sys
import logging

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from .prompt_control.node_clip import EditableCLIPEncode, ScheduleToCond
from .prompt_control.node_lora import LoRAScheduler, ScheduleToModel, PCSplitSampling
from .prompt_control.node_other import (
    PromptToSchedule,
    FilterSchedule,
)
from .prompt_control.node_aio import PromptControlSimple


NODE_CLASS_MAPPINGS = {
    "PromptControlSimple": PromptControlSimple,
    "PromptToSchedule": PromptToSchedule,
    "PCSplitSampling": PCSplitSampling,
    "FilterSchedule": FilterSchedule,
    "ScheduleToCond": ScheduleToCond,
    "ScheduleToModel": ScheduleToModel,
    "EditableCLIPEncode": EditableCLIPEncode,
    "LoRAScheduler": LoRAScheduler,
}
