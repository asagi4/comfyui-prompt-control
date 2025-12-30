"""
@author: asagi4
@title: ComfyUI Prompt Control
@nickname: ComfyUI Prompt Control
@description: Control LoRA and prompt scheduling, advanced text encoding, regional prompting, and much more, through your text prompt. Generates dynamic graphs that are literally identical to handcrafted noodle soup.
"""

import logging
import os
import sys

log = logging.getLogger("comfyui-prompt-control")

if os.environ.get("PROMPTCONTROL_DEBUG"):
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)

WEB_DIRECTORY = "web"

v1_modules = []
v3_modules = []
# Importing things here breaks pytest for whatever reason...
if "PYTEST_CURRENT_TEST" not in os.environ:
    import importlib

    from comfy_api.latest import ComfyExtension

    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[PromptControl] %(levelname)s: %(message)s"))
        log.addHandler(h)
    for node in ["base", "hooks", "tools", "lazy"]:
        mod = importlib.import_module(f".prompt_control.nodes_{node}", package=__name__)
        v3_modules.append(mod)

    class PromptControlExtension(ComfyExtension):
        async def get_node_list(self):
            r = []
            for m in v3_modules:
                r.extend(m.NODES)
            return r

    async def comfy_entrypoint():
        return PromptControlExtension()
