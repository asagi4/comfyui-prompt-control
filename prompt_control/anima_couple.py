# Adapted from https://github.com/pamparamm/ComfyUI-ppm
import torch
from comfy.ldm.anima.model import Anima as AnimaDIT
from comfy.patcher_extension import WrapperExecutor
from comfy.sampler_helpers import convert_cond
from comfy.samplers import process_conds


def anima_sample_wrapper(executor, *args, **kwargs):
    guider, _, extra_options, _, noise, latent_image, denoise_mask, *_ = args
    seed = extra_options["seed"]
    device = "cuda"  # TODO: fix

    def pc_process_conds(pc_conds):
        conds = [convert_cond([c])[0] for c in pc_conds]
        conds = process_conds(
            guider.inner_model,
            noise,
            {"positive": conds},
            device,
            latent_image,
            denoise_mask,
            seed,
            latent_shapes=[latent_image.shape],
        )
        return [
            c["model_conds"]["c_crossattn"].cond * pc_conds[i][1].get("strength", 1.0)
            for i, c in enumerate(conds["positive"])
        ]

    extra_options["model_options"]["transformer_options"]["pc_process_conds"] = pc_process_conds
    return executor(*args, **kwargs)


def anima_forward_wrapper(executor: WrapperExecutor, *args, **kwargs):
    """Model wrapper does something with activation shapes?"""
    anima_model: AnimaDIT = executor.class_obj  # type: ignore

    x: torch.Tensor = args[0]
    transformer_options: dict = kwargs.get("transformer_options", {}).copy()
    pc = transformer_options.get("pc_couple")
    if pc and "processed_conds" not in pc:
        pc["processed_conds"] = transformer_options["pc_process_conds"](pc["conds"])
    patch_spatial = anima_model.patch_spatial

    activations_shape = list(x.shape)
    activations_shape[-2] = activations_shape[-2] // patch_spatial
    activations_shape[-1] = activations_shape[-1] // patch_spatial

    transformer_options["activations_shape"] = activations_shape
    kwargs["transformer_options"] = transformer_options

    return executor(*args, **kwargs)
