"""
@author: asagi4
@title: ComfyUI Prompt Control
@nickname: ComfyUI Prompt Control
@description: Control LoRA and prompt scheduling, advanced text encoding, regional prompting, and much more, through your text prompt. Generates dynamic graphs that are literally identical to handcrafted noodle soup.
"""

import importlib
import logging
import os
import sys
from dataclasses import dataclass

from comfy_api.latest import ComfyExtension

log = logging.getLogger("comfyui-prompt-control")
if not log.handlers and "PYTEST_CURRENT_TEST" not in os.environ:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[PromptControl] %(levelname)s: %(message)s"))
    log.addHandler(h)

if os.environ.get("PROMPTCONTROL_DEBUG"):
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)

WEB_DIRECTORY = "web"

nodes = ["lazy", "tools", "hooks"]
v3_nodes = ["base"]
if "PYTEST_CURRENT_TEST" in os.environ:
    nodes = []

v1_modules = []
for node in nodes:
    mod = importlib.import_module(f".prompt_control.nodes_{node}", package=__name__)
    v1_modules.append(mod)

v3_modules = []
for node in v3_nodes:
    mod = importlib.import_module(f".prompt_control.nodes_{node}", package=__name__)
    v1_modules.append(mod)


@dataclass
class PCSchemaStub:
    node_id: str
    display_name: str


def v3_stub(module):
    def inject(cls, node_id, display_name):
        schema = PCSchemaStub(node_id, display_name)
        if not hasattr(cls, "GET_SCHEMA"):
            setattr(cls, "GET_SCHEMA", lambda: schema)
        return cls

    return [
        inject(cls, name, module.NODE_DISPLAY_NAME_MAPPINGS[name]) for name, cls in module.NODE_CLASS_MAPPINGS.items()
    ]


class PromptControlExtension(ComfyExtension):
    async def get_node_list(self):
        r = []
        for m in v1_modules:
            r.extend(v3_stub(m))
        for m in v3_modules:
            r.extend(m.NODES)
        return r


async def comfy_entrypoint():
    return PromptControlExtension()
