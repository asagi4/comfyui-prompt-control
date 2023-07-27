from pathlib import Path

import logging
from os import environ

log = logging.getLogger("prompt-control")
logging.basicConfig()
if environ.get("COMFY_PC_DEBUG", False):
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)

import time

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.sd
import folder_paths


def getlogger():
    return log


def load_loras_from_schedule(schedules, loaded_loras):
    lora_specs = []
    for step, sched in schedules:
        if sched["loras"]:
            lora_specs.extend(sched["loras"])
    loaded_loras = load_loras(lora_specs)
    return loaded_loras


class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        log.debug(f"Executed {self.name} in {elapsed} seconds")


def schedule_for_step(total_steps, step, schedules):
    for end, s in schedules:
        if end * total_steps > step:
            return [end, s]
    return schedules[-1]


def load_loras(lora_specs, loaded_loras=None):
    loaded_loras = loaded_loras or {}
    filenames = [Path(f) for f in folder_paths.get_filename_list("loras")]
    names = set(name for name, _ in lora_specs)
    for name in names:
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
