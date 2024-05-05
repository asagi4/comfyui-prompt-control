from collections import namedtuple
from os import environ
from pathlib import Path
import re
from math import lcm
import time
import logging
import torch


import nodes
import folder_paths

import comfy.model_management

log = logging.getLogger("comfyui-prompt-control")

FORCE_CPU_OFFLOAD = bool(environ.get("COMFYUI_PC_CPU_OFFLOAD"))
CACHE_MODELS = bool(environ.get("COMFYUI_PC_CACHE_MODEL", False))
CACHED_MODEL = None
CACHED_CLONE = None


def finish_sampling(model):
    # Hold on to the current model state in case the next gen
    # doesn't need to apply more LoRAs
    if not CACHE_MODELS:
        unpatch_model(model)
    log.info("Preventing ComfyUI model unpatch")
    model.backup = {}


def set_state(model, key, value):
    s = get_state(model, None)
    s[key] = value


# Lambda avoids deepcopy
def get_state(model, key):
    if "pc_model_state" in model.model_options:
        s = model.model_options["pc_model_state"]()
        if key is None:
            return s
        else:
            return s.get(key)
    # Init missing state and retry
    x = dict(applied_loras={})
    model.model_options["pc_model_state"] = lambda: x
    return get_state(model, key)


def get_cached_model(model):
    global CACHED_MODEL
    global CACHED_CLONE
    if not CACHE_MODELS:
        return clone_model(model)

    if model != CACHED_MODEL:
        if CACHED_CLONE is not None:
            unpatch_model(CACHED_CLONE)
        CACHED_CLONE = clone_model(model)
        # Make sure the state exists
        set_state(CACHED_CLONE, "backup", CACHED_CLONE.backup)
        CACHED_MODEL = model
    # double clone maintains LoRA state because it's shared between clones, but allows other modifications without mixing things up
    model = clone_model(CACHED_CLONE)
    return CACHED_CLONE


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


def get_function(text, func, defaults, return_func_name=False):
    rex = re.compile(rf"\b{func}\(", re.MULTILINE)
    instances = []
    match = rex.search(text)
    while match:
        # Match start, content start
        start, after_first_paren = match.span()
        funcname = text[start : after_first_paren - 1]
        end = find_closing_paren(text, after_first_paren)
        args = parse_strings(text[after_first_paren:end], defaults)
        if return_func_name:
            instances.append((funcname, args))
        else:
            instances.append(args)

        text = text[:start] + text[end + 1 :]
        match = rex.search(text)
    return text, instances


def parse_args(strings, arg_spec, strip=True):
    args = [s[1] for s in arg_spec]
    for i, spec in list(enumerate(arg_spec))[: len(strings)]:
        try:
            if strip:
                strings[i] = strings[i].strip()
            args[i] = spec[0](strings[i])
        except ValueError:
            pass
    return args


def parse_floats(string, defaults, split_re=","):
    spec = [(float, d) for d in defaults]
    return parse_args(re.split(split_re, string.strip()), spec)


def parse_strings(string, defaults, split_re=r"(?<!\\),", replace=(r"\,", ",")):
    if defaults is None:
        return string
    spec = [(lambda x: x, d) for d in defaults]
    splits = re.split(split_re, string)
    if replace:
        f, t = replace
        splits = [s.replace(f, t) for s in splits]
    return parse_args(splits, spec, strip=False)


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


def unpatch_model(model):
    if model:
        if "pc_model_state" in model.model_options:
            s = model.model_options["pc_model_state"]()
            if "backup" in s:
                for k in s["backup"]:
                    if k not in model.backup:
                        model.backup[k] = s["backup"][k]
        log.info("Unpatching model")
        model.unpatch_model()
        set_state(model, "backup", model.backup)
        set_state(model, "applied_loras", {})


def clone_model(model):
    if not model:
        return None
    model = model.clone()
    if not environ.get("PC_NO_INPLACE_UPDATE"):
        model.weight_inplace_update = True
    return model


def add_patches(model, patches, weight):
    model.add_patches(patches, weight)


def patch_model(model, forget=False, orig=None):
    global FORCE_CPU_OFFLOAD
    try:
        return _patch_model(model, forget, orig, FORCE_CPU_OFFLOAD)
    except comfy.model_management.OOM_EXCEPTION:
        FORCE_CPU_OFFLOAD = True
        log.error("Ran out of memory while applying LoRAs, Forcing CPU offload from now on")
        # Unpatch to restore partially applied weights
        unpatch_model(model)
        raise


def _patch_model(model, forget=False, orig=None, offload_to_cpu=False):
    if not model:
        return None
    if offload_to_cpu:
        saved_offload = model.offload_device
        model.offload_device = torch.device("cpu")
    log.info("Patching model, cpu_offload=%s", model.offload_device == torch.device("cpu"))
    if orig:
        model.backup = orig.backup
    model.patch_model()
    if offload_to_cpu:
        model.offload_device = saved_offload
    if forget:
        model.patches = {}
        model.object_patches = {}
    return model


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
    rootlogger = logging.getLogger()
    oldlevel = rootlogger.level
    try:
        rootlogger.setLevel(logging.ERROR)
        x = f()
    except BaseException:
        __builtins__["print"] = p
        rootlogger.setLevel(oldlevel)
        raise
    __builtins__["print"] = p
    rootlogger.setLevel(oldlevel)
    return x


def lora_name_to_file(name):
    filenames = folder_paths.get_filename_list("loras")
    # Return exact matches as is
    if name in filenames:
        return name
    # Some autocompletion scripts replace _ with spaces
    for n in [name, name.replace(" ", "_")]:
        for f in filenames:
            p = Path(f).with_suffix("")
            if p.name == n or str(p) == n:
                return f
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


def apply_loras_from_spec(
    loraspec, model=None, clip=None, orig_model=None, orig_clip=None, patch=False, cache=None, applied_loras=None
):
    if applied_loras is None:
        applied_loras = {}
    actual_loraspec = {}
    additive = True
    for key in loraspec:
        if key in applied_loras and applied_loras[key] == loraspec[key]:
            continue
        if key in applied_loras and applied_loras[key] != loraspec[key]:
            additive = False
        actual_loraspec[key] = loraspec[key]

    for key in applied_loras:
        if key not in loraspec:
            actual_loraspec = loraspec
            additive = False

    backup_model = model
    if not additive:
        unpatch_model(model)
        # Reset clip to unpatched
        if clip:
            clip = orig_clip or clip

    if cache is None:
        cache = {}
    if not loraspec:
        return model, clip

    for name, params in actual_loraspec.items():
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
                log.warning("Lora %s not found", name)
                continue
            log.info("Loading LoRA: %s", f)
            loader = make_loader(f, bool(lbw))
            cache[cache_key] = loader

        m, c = loader(m, c, w, w_clip, lbw)
        model = m or model
        clip = c or clip
        if model:
            log.info("Applying LoRA: %s:%s, LBW=%s, additive=%s", name, params["weight"], bool(lbw), additive)
        if clip:
            log.info("Applying CLIP LoRA: %s:%s, LBW=%s, additive=%s", name, params["weight_clip"], bool(lbw), additive)

    # forget patches so we don't double-patch
    model = patch_model(model, forget=True, orig=backup_model)
    return model, clip


class Timer:
    def __init__(self, name):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if environ.get("PC_SHOW_TIMINGS"):
            log.info("Executed %s in %s seconds", self.name, elapsed)
