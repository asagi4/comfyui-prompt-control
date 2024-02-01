from .utils import get_callback, unpatch_model
import sys

import logging
import gc
import comfy.model_management
import os

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

    from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler

    def pc_sample(*args, **kwargs):
        model = args[0]
        cb = get_callback(model)
        BrownianTreeNoiseSampler.pc_reset(
            model.model_options.get("pc_split_sampling"),
            kwargs.get("force_full_denoise") or kwargs.get("denoise", 1.0) >= 1.0,
        )
        if cb:
            try:
                try:
                    r = cb(orig_sampler, *args, **kwargs)
                except comfy.model_management.OOM_EXCEPTION:
                    if not os.environ.get("PC_RETRY_ON_OOM"):
                        raise
                    log.error("Got OOM while sampling, freeing memory and retrying once...")
                    unpatch_model(model)
                    BrownianTreeNoiseSampler.pc_reset(False)
                    gc.collect()
                    comfy.model_management.soft_empty_cache()
                    r = cb(orig_sampler, *args, **kwargs)
            except Exception:
                log.error("Exception occurred during callback, unpatching model.")
                unpatch_model(model)
                BrownianTreeNoiseSampler.pc_reset(False)
                raise
        else:
            r = orig_sampler(*args, **kwargs)
        BrownianTreeNoiseSampler.pc_reset()
        return r

    hijack(mod, function, pc_sample)


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

            from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler

            BrownianTreeNoiseSampler.set_global_sigmas(self.sigmas)

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


def hijack_browniannoisesampler(module, cls):
    mod = sys.modules[module]
    orig_sampler = getattr(mod, cls)
    if has_hijack(orig_sampler):
        return

    class PCBrownianTreeNoiseSampler(orig_sampler):
        global_instance = None
        use_global_sigmas = False
        global_sigmas = None
        force_full_denoise = False

        @classmethod
        def pc_reset(cls, use_global_sigmas=False, force_full_denoise=False):
            cls.global_instance = None
            cls.global_sigmas = None
            cls.use_global_sigmas = use_global_sigmas
            cls.force_full_denoise = force_full_denoise

        @classmethod
        def set_global_sigmas(cls, sigmas):
            if cls.global_sigmas is None and cls.use_global_sigmas:
                cls.global_sigmas = (0 if cls.force_full_denoise else sigmas[sigmas > 0].min(), sigmas.max())
                log.info(
                    "Initializing BrownianTreeNoiseSampler instance with global sigmas %s, %s",
                    cls.global_sigmas,
                    cls.force_full_denoise,
                )

        def __init__(self, x, sigma_min, sigma_max, **kwargs):
            if self.global_sigmas is not None:
                sigma_min, sigma_max = self.global_sigmas
            if not self.global_instance:
                super().__init__(x, sigma_min, sigma_max, **kwargs)
                PCBrownianTreeNoiseSampler.global_instance = self

        def __call__(self, *args, **kwargs):
            if self.global_instance and self != self.global_instance:
                return self.global_instance(*args, **kwargs)
            else:
                return super().__call__(*args, **kwargs)

    hijack(mod, cls, PCBrownianTreeNoiseSampler)


def do_hijack():
    hijack_browniannoisesampler("comfy.k_diffusion.sampling", "BrownianTreeNoiseSampler")
    hijack_sampler("comfy.sample", "sample")
    hijack_sampler("comfy.sample", "sample_custom")
    hijack_ksampler("comfy.samplers", "KSampler")
