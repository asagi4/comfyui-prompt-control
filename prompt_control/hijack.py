from .utils import get_callback, unpatch_model
import sys

import logging

log = logging.getLogger("comfyui-prompt-control")


def has_hijack(obj):
    return hasattr(obj, "pc_hijack_done")


def hijack(obj, attr, replacement):
    setattr(obj, attr, replacement)
    setattr(replacement, "pc_hijack_done", True)


def hijack_sampler(module, function):
    mod = sys.modules[module]
    orig_sampler = getattr(mod, function)
    if has_hijack(orig_sampler):
        return

    def pc_sample(*args, **kwargs):
        model = args[0]
        cb = get_callback(model)
        if cb:
            try:
                return cb(orig_sampler, *args, **kwargs)
            except Exception:
                log.info("Exception occurred during callback, unpatching model...")
                unpatch_model(model)
                raise
        else:
            return orig_sampler(*args, **kwargs)

    hijack(mod, function, pc_sample)


def hijack_ksampler(module, cls):
    mod = sys.modules[module]
    orig_sampler = getattr(mod, cls)
    if has_hijack(orig_sampler):
        return

    class HijackedKSampler(orig_sampler):
        def sample(self, *args, **kwargs):
            log.info("Hijacked ksampler call")
            return super().sample(*args, **kwargs)

    hijack(mod, cls, HijackedKSampler)


def hijack_aitemplate():
    try:
        AITLoader = sys.modules["AIT.AITemplate.ait.load"].AITLoader
        orig_apply_unet = AITLoader.apply_unet
        if has_hijack(orig_apply_unet):
            return
        log.info("AITemplate detected, hijacking...")

        def apply_unet(self, *args, **kwargs):
            setattr(self, "pc_applied_module", kwargs["aitemplate_module"])
            return orig_apply_unet(self, *args, **kwargs)

        hijack(AITLoader, "apply_unet", apply_unet)
        log.info("AITemplate hijack complete")
    except Exception:
        log.info("AITemplate hijack failed or not necessary")


# Default hijack. No KSampler hijack for now
def do_hijack():
    hijack_sampler("comfy.sample", "sample")
    hijack_aitemplate()
