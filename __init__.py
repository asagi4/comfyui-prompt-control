import os
import sys
import logging
import importlib


log = logging.getLogger("comfyui-prompt-control")
log.propagate = False
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(levelname)s] PromptControl: %(message)s"))
    log.addHandler(h)

if os.environ.get("PROMPTCONTROL_DEBUG"):
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

nodes = ['base', 'lazy']
if importlib.util.find_spec("comfy.hooks"):
    nodes.append('hook')
else:
    log.warning(
        "Your ComfyUI version is too old, can't import comfy.hooks for PCEncodeSchedule and PCLoraHooksFromSchedule. Update your installation."
    )

for node in nodes:
    mod = importlib.import_module(f".prompt_control.nodes_{node}", package=__name__)
    NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
