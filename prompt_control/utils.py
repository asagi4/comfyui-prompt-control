from pathlib import Path
from math import lcm
import time
import re

import folder_paths

import logging
import sys
from os import environ
from collections import namedtuple
import nodes

log = logging.getLogger("comfyui-prompt-control")


# Minimal Modelpatcher that doesn't do anything, for LoRA loading when not
# interested in either CLIP or unet
class DummyModelPatcher:
    class DummyTorchModel:
        def __init__(self):
            dummyconf = {
                "num_res_blocks": [],
                "channel_mult": [],
                "transformer_depth": [],
                "transformer_depth_output": [],
                "transformer_depth_middle": 0,
            }
            self.model_config = namedtuple("DummyConfig", ["unet_config"])(dummyconf)

        def state_dict(self):
            return {}

    def __init__(self):
        self.model = self.DummyTorchModel()
        self.cond_stage_model = self.DummyTorchModel()
        self.weight_inplace_update = True
        self.model_options = {}

    def add_patches(self, patches, *args, **kwargs):
        return []

    def patch_model(self):
        pass

    def unpatch_model(self):
        pass

    def clone(self):
        return self


DUMMY_MODEL = DummyModelPatcher()


def find_closing_paren(text, start):
    stack = 1
    for i, char in enumerate(text[start:]):
        if char == ")":
            stack -= 1
        elif char == "(":
            stack += 1
        if stack == 0:
            return start + i
    # Implicit closing paren after end
    return len(text)


def get_function(text, func, defaults):
    rex = re.compile(rf"\b{func}\(", re.MULTILINE)
    instances = []
    match = rex.search(text)
    while match:
        # Match start, content start
        start, after_first_paren = match.span()
        end = find_closing_paren(text, after_first_paren)
        instances.append(text[after_first_paren:end])
        text = text[:start] + text[end + 1 :]
        match = rex.search(text)
    return text, [parse_strings(i, defaults) for i in instances]


def parse_args(strings, arg_spec):
    args = [s[1] for s in arg_spec]
    for i, spec in list(enumerate(arg_spec))[: len(strings)]:
        try:
            args[i] = spec[0](strings[i].strip())
        except ValueError:
            pass
    return args


def parse_floats(string, defaults, split_re=","):
    spec = [(float, d) for d in defaults]
    return parse_args(re.split(split_re, string.strip()), spec)


def parse_strings(string, defaults, split_re=","):
    if defaults is None:
        return string
    spec = [(lambda x: x, d) for d in defaults]
    return parse_args(re.split(split_re, string.strip()), spec)


def equalize(*tensors):
    if all(t.shape[1] == tensors[0].shape[1] for t in tensors):
        return tensors

    x = lcm(*(t.shape[1] for t in tensors))

    return (t.repeat(1, x // t.shape[1], 1) for t in tensors)


def safe_float(f, default):
    if f is None:
        return default
    try:
        return round(float(f), 2)
    except ValueError:
        return default


def get_aitemplate_module():
    return sys.modules["AIT.AITemplate.AITemplate"]


def unpatch_model(model):
    if model:
        model.unpatch_model()


def clone_model(model):
    if not model:
        return None
    model = model.clone()
    if not environ.get("PC_NO_INPLACE_UPDATE"):
        model.weight_inplace_update = True
    return model


def add_patches(model, patches, weight):
    model.add_patches(patches, weight)


def patch_model(model):
    if not model:
        return None
    if "aitemplate_keep_loaded" in model.model_options:
        model.patch_model()
        mod = get_aitemplate_module()
        l = mod.AITemplate.loader
        if hasattr(l, "pc_applied_module"):
            log.info("Applying AITemplate unet")
            l.apply_unet(
                aitemplate_module=l.pc_applied_module,
                unet=l.compvis_unet(model.model.state_dict()),
                in_channels=model.model.diffusion_model.in_channels,
                conv_in_key="conv_in_weight",
            )
    else:
        model.patch_model()


def get_callback(model):
    return model.model_options.get("prompt_control_callback")


def set_callback(model, cb):
    model.model_options["prompt_control_callback"] = cb


# Hack to temporarily override printing to stdout to stop log spam
def suppress_print(f):
    def noop(*args):
        pass

    p = print
    __builtins__["print"] = noop
    try:
        x = f()
    except:
        __builtins__["print"] = p
        raise
    __builtins__["print"] = p
    return x


def lora_name_to_file(name):
    filenames = [Path(f) for f in folder_paths.get_filename_list("loras")]
    for f in filenames:
        if f.stem == name:
            return str(f)
    log.warning("Lora %s not found", name)
    return None


def load_lbw():
    return nodes.NODE_CLASS_MAPPINGS.get("LoraLoaderBlockWeight //Inspire")


def make_loader(filename, lbw):
    if not lbw:
        l = nodes.LoraLoader()

        def loader(model, clip, model_weight, clip_weight, lbw):
            return suppress_print(lambda: l.load_lora(model, clip, filename, model_weight, clip_weight))

    else:
        # This is already checked before calling make_loader
        l = load_lbw()()

        def loader(model, clip, model_weight, clip_weight, lbw):
            spec = lbw["LBW"]
            lbw_a = safe_float(lbw.get("A"), 4.0)
            lbw_b = safe_float(lbw.get("B"), 1.0)
            m = model or DUMMY_MODEL
            c = clip or DUMMY_MODEL
            m, c, _ = suppress_print(
                lambda: l.doit(m, c, filename, model_weight, clip_weight, False, 0, lbw_a, lbw_b, "", spec)
            )
            if m is DUMMY_MODEL:
                m = None
            if c is DUMMY_MODEL:
                c = None
            return m, c

    return loader


def apply_loras_from_spec(loraspec, model=None, clip=None, orig_model=None, orig_clip=None, patch=False, cache=None):
    if patch:
        unpatch_model(model)
        model = clone_model(orig_model or model)
        unpatch_model(clip)
        clip = clone_model(orig_clip or clip)

    if cache is None:
        cache = {}
    if not loraspec:
        return model, clip

    for name, params in loraspec.items():
        m, c = model, clip
        w, w_clip = params["weight"], params["weight_clip"]
        if w == 0:
            m = None
        if w_clip == 0:
            c = None
        if not w and not c:
            continue

        lbw = params.get("lbw")
        if lbw and not load_lbw():
            log.warning("LoraBlockWeight not available, ignoring LBW parameters")
            lbw = None

        # Cache the loader instance so that it doesn't reload the LoRA from disk all the time
        cache_key = name, bool(lbw)
        loader = cache.get(cache_key)
        if not loader:
            f = lora_name_to_file(name)
            if not f:
                continue
            loader = make_loader(f, bool(lbw))
            cache[cache_key] = loader

        m, c = loader(m, c, w, w_clip, lbw)
        model = m or model
        clip = c or clip
        if model:
            log.info("LoRA applied: %s:%s, LBW=%s", name, params["weight"], bool(lbw))
        if clip:
            log.info("CLIP LoRA applied: %s:%s, LBW=%s", name, params["weight_clip"], bool(lbw))
    if patch:
        patch_model(model)
        patch_model(clip)
    return model, clip


class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if environ.get("COMFYUI_PC_TIMING"):
            log.info(f"Executed {self.name} in {elapsed} seconds")
