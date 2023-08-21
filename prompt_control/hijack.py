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


def calculate_absolute_timesteps(model, conds, sigmas):
    def find_sigma_t(pct):
        idx = int(round((1 - pct) * (len(sigmas) - 1), 0))
        s = model.sigma_to_t(sigmas[idx])
        return s

    for t in range(len(conds)):
        c = conds[t]
        if not c[1].get("absolute_timesteps"):
            continue

        timestep_start = None
        timestep_end = None
        if "start_percent" in c[1]:
            timestep_start = find_sigma_t(c[1]["start_percent"])
        if "end_percent" in c[1]:
            timestep_end = find_sigma_t(c[1]["end_percent"])
        n = c[1].copy()

        if timestep_start:
            del n["start_percent"]
            n["timestep_start"] = timestep_start
        if timestep_end:
            del n["end_percent"]
            n["timestep_end"] = timestep_end
        conds[t] = [c[0], n]


def hijack_ksampler(module, cls):
    mod = sys.modules[module]
    orig_sampler = getattr(mod, cls)
    if has_hijack(orig_sampler):
        return

    class HijackedKSampler(orig_sampler):
        def sample(
            self,
            noise,
            positive,
            negative,
            cfg,
            latent_image=None,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            denoise_mask=None,
            sigmas=None,
            callback=None,
            disable_pbar=False,
            seed=None,
        ):
            if sigmas is None:
                sigmas = self.sigmas
            if last_step is not None and last_step < (len(sigmas) - 1):
                sigmas = sigmas[: last_step + 1]
            if start_step is not None and start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step]

            calculate_absolute_timesteps(self.model_wrap, positive, sigmas)
            calculate_absolute_timesteps(self.model_wrap, negative, sigmas)

            return super().sample(
                noise,
                positive,
                negative,
                cfg,
                latent_image,
                start_step,
                last_step,
                force_full_denoise,
                denoise_mask,
                sigmas,
                callback,
                disable_pbar,
                seed,
            )

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
    hijack_ksampler("comfy.samplers", "KSampler")
