import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from .prompt_control.nodes import NODE_CLASS_MAPPINGS
