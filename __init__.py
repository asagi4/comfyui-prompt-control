"""
@author: asagi4
@title: ComfyUI Prompt Control
@nickname: ComfyUI Prompt Control
@description: Control LoRA and prompt scheduling, advanced text encoding, regional prompting, and much more, through your text prompt. Generates dynamic graphs that are literally identical to handcrafted noodle soup.
"""

import os
import sys
import logging
import importlib


log = logging.getLogger("comfyui-prompt-control")
log.propagate = False
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[PromptControl] %(levelname)s: %(message)s"))
    log.addHandler(h)

if os.environ.get("PROMPTCONTROL_DEBUG"):
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "web"

nodes = ["base", "lazy", "tools", "hooks"]

for node in nodes:
    mod = importlib.import_module(f".prompt_control.nodes_{node}", package=__name__)
    NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
