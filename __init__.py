import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from .nodes import EditableCLIPEncode, LoRAScheduler
from .utils import getlogger
import comfy.sample

log = getlogger()

if not getattr(comfy.sample.sample, 'prompt_control_monkeypatch', False):
    log.info("Monkeypatching comfy.sample.sample...")
    orig_sample = comfy.sample.sample
    def sample(*args, **kwargs):
        model = args[0]
        if hasattr(model, 'prompt_control_callback'):
            args, kwargs = model.prompt_control_callback(*args, **kwargs)
        return orig_sample(*args, **kwargs)
    setattr(sample, 'prompt_control_monkeypatch', True)
    comfy.sample.sample = sample

NODE_CLASS_MAPPINGS = {
    "EditableCLIPEncode": EditableCLIPEncode,
    "LoRAScheduler": LoRAScheduler,
}

