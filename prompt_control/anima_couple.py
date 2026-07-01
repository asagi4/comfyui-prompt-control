# Adapted from https://github.com/pamparamm/ComfyUI-ppm
import itertools
from collections.abc import Callable
from functools import partial
from math import lcm

import torch
import torch.nn.functional as F
from comfy.ldm.anima.model import Anima as AnimaDIT
from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
from comfy.patcher_extension import WrapperExecutor
from comfy.sampler_helpers import convert_cond
from comfy.samplers import process_conds

COND = 0
UNCOND = 1


def reshape_mask(mask: torch.Tensor, size: tuple[int, int], bs: int, num_tokens: int) -> torch.Tensor:
    num_conds = mask.shape[0]

    mask_downsample = F.interpolate(mask, size=size, mode="nearest")
    mask_downsample_reshaped = mask_downsample.view(num_conds, num_tokens, 1).repeat_interleave(bs, dim=0)

    return mask_downsample_reshaped


def wrap_forwards(anima_model):
    backups = {}
    for block_name, b in (
        (n, b) for n, b in anima_model.named_modules() if "cross_attn" in n and isinstance(b, CosmosAttention)
    ):
        backups[block_name] = b.forward
        b.forward = partial(cosmos_attention_forward_couple, b.forward)
    return backups


def unwrap_forwards(anima_model, backups):
    for block_name, b in (
        (n, b) for n, b in anima_model.named_modules() if "cross_attn" in n and isinstance(b, CosmosAttention)
    ):
        b.forward = backups[block_name]


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

    b = {}
    if pc:
        b = wrap_forwards(anima_model)
    r = executor(*args, **kwargs)
    if pc:
        unwrap_forwards(anima_model, b)
    return r


def cosmos_attention_forward_couple(_forward: Callable, x, context, rope_emb, transformer_options):
    """attention block wrapper"""
    if "pc_couple" not in transformer_options:
        return _forward(x, context, rope_emb, transformer_options)
    c: torch.Tensor = context
    # FIXME: base cond weight
    # c = args["processed_conds"][0]

    args = transformer_options["pc_couple"]

    mask = args["mask"]
    conds = args["processed_conds"][1:]
    num_conds = len(conds) + 1
    num_tokens_c: list[int] = [c.shape[1] for c in conds]
    cond_or_uncond = transformer_options["cond_or_uncond"]
    cond_or_uncond_couple = []

    num_chunks = len(cond_or_uncond)
    bs = x.shape[0] // num_chunks

    x_chunks = x.chunk(num_chunks, dim=0)
    c_chunks = c.chunk(num_chunks, dim=0)
    lcm_tokens_c = lcm(c.shape[1], *num_tokens_c)
    conds_c_tensor = torch.cat(
        [cond.repeat(bs, lcm_tokens_c // num_tokens_c[i], 1) for i, cond in enumerate(conds)],
        dim=0,
    )

    xs, cs = [], []
    for i, cond_type in enumerate(cond_or_uncond):
        x_target = x_chunks[i]
        c_target = c_chunks[i].repeat(1, lcm_tokens_c // c.shape[1], 1)
        if cond_type == UNCOND:
            xs.append(x_target)
            cs.append(c_target)
            cond_or_uncond_couple.append(UNCOND)
        else:
            xs.append(x_target.repeat(num_conds, 1, 1))
            cs.append(torch.cat([c_target, conds_c_tensor], dim=0))
            cond_or_uncond_couple.extend(itertools.repeat(COND, num_conds))

    xs = torch.cat(xs, dim=0)
    cs = torch.cat(cs, dim=0)

    out = _forward(xs, cs, rope_emb, transformer_options)

    size = tuple(transformer_options["activations_shape"][-2:])
    num_tokens = out.shape[1]
    mask_downsample = reshape_mask(mask, size, bs, num_tokens)

    outputs = []
    cond_outputs = []
    i_cond = 0

    for i, cond_type in enumerate(cond_or_uncond_couple):
        pos, next_pos = i * bs, (i + 1) * bs

        if cond_type == UNCOND:
            outputs.append(out[pos:next_pos])
        else:
            pos_cond, next_pos_cond = i_cond * bs, (i_cond + 1) * bs
            masked_output = out[pos:next_pos] * mask_downsample[pos_cond:next_pos_cond]
            cond_outputs.append(masked_output)
            i_cond += 1

    if len(cond_outputs) > 0:
        cond_output = torch.stack(cond_outputs).sum(0)
        outputs.append(cond_output)

    return torch.cat(outputs, dim=0)
