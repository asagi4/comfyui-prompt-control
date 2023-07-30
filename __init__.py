import os
import sys
import logging

log = logging.getLogger('comfyui-prompt-control')
logging.basicConfig()
if os.environ.get('COMFYUI_PC_DEBUG'):
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from .prompt_control.node_clip import EditableCLIPEncode
from .prompt_control.node_lora import LoRAScheduler
from .prompt_control.node_other import ConditioningCutoff

NODE_CLASS_MAPPINGS = {
    "EditableCLIPEncode": EditableCLIPEncode,
    "LoRAScheduler": LoRAScheduler,
    "ConditioningCutoff": ConditioningCutoff,
}
