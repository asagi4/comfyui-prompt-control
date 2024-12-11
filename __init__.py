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


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

from .prompt_control.node_other import NODE_CLASS_MAPPINGS as o_mappings, NODE_DISPLAY_NAME_MAPPINGS as o_display
from .prompt_control.nodes_lazy import NODE_CLASS_MAPPINGS as lazy_mappings, NODE_DISPLAY_NAME_MAPPINGS as lazy_display

NODE_CLASS_MAPPINGS.update(o_mappings)
NODE_CLASS_MAPPINGS.update(lazy_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(o_display)
NODE_DISPLAY_NAME_MAPPINGS.update(lazy_display)

import importlib

if importlib.util.find_spec("comfy.hooks"):
    from .prompt_control.nodes_hooks import (
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
