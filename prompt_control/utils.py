from pathlib import Path
from math import lcm
import time
import re

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.sd
import comfy.lora
import folder_paths

import logging
import sys
from os import environ

log = logging.getLogger("comfyui-prompt-control")


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
    try:
        return round(float(f), 2)
    except ValueError:
        return default


def get_aitemplate_module():
    return sys.modules["AIT.AITemplate.AITemplate"]


def unpatch_model(model):
    model.unpatch_model()


def clone_model(model):
    model = model.clone()
    if environ.get("PC_INPLACE_UPDATE"):
        model.weight_inplace_update = True
    return model


def add_patches(model, patches, weight):
    model.add_patches(patches, weight)


def patch_model(model):
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


def get_lora_keymap(model=None, clip=None):
    key_map = {}
    if model:
        key_map = comfy.lora.model_lora_keys_unet(model.model)
    if clip:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    return key_map


def load_lora(model, lora, weight, key_map, clone=True):
    # Hack to temporarily override printing to stdout to stop log spam
    def noop(*args):
        pass

    p = print
    __builtins__["print"] = noop
    loaded = comfy.lora.load_lora(lora, key_map)
    __builtins__["print"] = p
    if clone:
        model = clone_model(model)
    add_patches(model, loaded, weight)
    return model


def apply_loras_to_model(model, orig_model, lora_specs, loaded_loras, patch=True):
    keymap = get_lora_keymap(model=model)
    if patch:
        unpatch_model(model)
        model = clone_model(orig_model)

    for name, params in lora_specs.items():
        if name not in loaded_loras or params["weight"] == 0:
            continue
        model = load_lora(model, loaded_loras[name], params["weight"], keymap)
        log.info("Loaded LoRA %s:%s", name, params["weight"])

    if patch:
        patch_model(model)

    return model


def load_loras_from_schedule(schedules, loaded_loras):
    lora_specs = {}
    for step, sched in schedules:
        if sched["loras"]:
            lora_specs.update(sched["loras"])
    loaded_loras = load_loras(lora_specs, loaded_loras)
    return loaded_loras


class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if environ.get("COMFYUI_PC_TIMING"):
            log.info(f"Executed {self.name} in {elapsed} seconds")


def load_loras(lora_specs, loaded_loras=None):
    loaded_loras = loaded_loras if loaded_loras is not None else {}
    filenames = [Path(f) for f in folder_paths.get_filename_list("loras")]
    for name in lora_specs.keys():
        if name in loaded_loras:
            continue
        found = False
        for f in filenames:
            if f.stem == name:
                full_path = folder_paths.get_full_path("loras", str(f))
                loaded_loras[name] = comfy.utils.load_torch_file(full_path, safe_load=True)
                found = True
                break
        if not found:
            log.warning("Lora %s not found", name)
    return loaded_loras
