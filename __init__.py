import os
import sys
import logging

log = logging.getLogger("comfyui-prompt-control")
logging.basicConfig()
if os.environ.get("COMFYUI_PC_DEBUG"):
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from .prompt_control.node_clip import EditableCLIPEncode, ScheduleToCond
from .prompt_control.node_lora import LoRAScheduler, ScheduleToModel
from .prompt_control.node_other import ConditioningCutoff, JinjaRender, PromptToSchedule, FilterSchedule, StringConcat
from .prompt_control.node_aio import PromptControlSimple


NODE_CLASS_MAPPINGS = {
    "PromptControlSimple": PromptControlSimple,
    "PromptToSchedule": PromptToSchedule,
    "FilterSchedule": FilterSchedule,
    "ScheduleToCond": ScheduleToCond,
    "ScheduleToModel": ScheduleToModel,
    "JinjaRender": JinjaRender,
    "StringConcat": StringConcat,
    "EditableCLIPEncode": EditableCLIPEncode,
    "LoRAScheduler": LoRAScheduler,
    "ConditioningCutoff": ConditioningCutoff,
}
