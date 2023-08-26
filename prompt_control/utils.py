from pathlib import Path
import time

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


def steps(start, end=None, step=0.1):
    if end is None:
        end = start
        start = step
    while start <= end:
        yield start
        start += step
        start = round(start, 2)


def get_aitemplate_module():
    return sys.modules["AIT.AITemplate.AITemplate"]


def unpatch_model(model):
    model.unpatch_model()


def clone_model(model):
        return model.clone()


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
                in_channels=m.model.diffusion_model.in_channels,
                conv_in_key="conv_in_weight",
            )
    else:
        model.patch_model(model.load_device)
        model.model.to(model.load_device)


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
