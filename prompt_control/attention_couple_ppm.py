# Lifted from https://github.com/pamparamm/ComfyUI-ppm/blob/c3e6b673ee2d424405dcb99aeed89f21943c89ac/nodes_ppm/attention_couple_ppm.py
# Original implementation by laksjdjf, hako-mikan, Haoming02 licensed under GPL-3.0
# https://github.com/laksjdjf/cgem156-ComfyUI/blob/1f5533f7f31345bafe4b833cbee15a3c4ad74167/scripts/attention_couple/node.py
# https://github.com/Haoming02/sd-forge-couple/blob/e8e258e982a8d149ba59a4bc43b945467604311c/scripts/attention_couple.py
import math

import torch
import torch.nn.functional as F
from comfy.hooks import TransformerOptionsHook, HookGroup, EnumHookScope, set_hooks_for_conditioning

from comfy.model_patcher import ModelPatcher

import logging

log = logging.getLogger("comfyui-prompt-control")

COND = 0
UNCOND = 1

DEBUG_KEYS = {}


def debug(key, message):
    if key not in DEBUG_KEYS:
        DEBUG_KEYS[key] = True
        print(key, message)


def set_cond_attnmask(base_cond, extra_conds, fill=False):
    hook = AttentionCoupleHook(base_cond[0], extra_conds, fill=fill)
    group = HookGroup()
    group.add(hook)
    return set_hooks_for_conditioning(base_cond, hooks=group)


def lcm_for_list(numbers):
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = math.lcm(current_lcm, number)
    return current_lcm


class Proxy:
    def __init__(self, function):
        self.function = function

    def to(self, *args, **kwargs):
        self.function.__self__.to(*args, **kwargs)
        return self

    def __call__(self, *args, **kwargs):
        return self.function(*args, *kwargs)


class AttentionCoupleHook(TransformerOptionsHook):
    def __init__(self, base_cond, conds, fill):
        DEBUG_KEYS.clear()
        super().__init__(hook_scope=EnumHookScope.HookedOnly)
        self.transformers_dict = {
            "patches": {
                "attn2_output_patch": [Proxy(self.attn2_output_patch)],
                "attn2_patch": [Proxy(self.attn2_patch)],
            }
        }

        self.conds_kv = []

        self.num_conds = len(conds) + 1
        self.base_strength = base_cond[1].pop("strength", 1.0)
        self.strengths = [cond[1].get("strength", 1.0) for cond in conds]
        self.conds: list[torch.Tensor] = [cond[0] for cond in conds]
        base_mask = base_cond[1].pop("mask", None)
        masks = [cond[1].pop("mask") * cond[1].pop("mask_strength") for cond in conds]

        if base_mask is None and not fill:
            raise ValueError("You must specify a base mask when fill=False")
        elif base_mask is None:
            sum = torch.stack(masks, dim=0).sum(dim=0)
            base_mask = torch.zeros_like(sum)
            base_mask[sum <= 0] = 1.0
        mask = [base_mask] + masks
        mask = torch.stack(mask, dim=0)
        if mask.sum(dim=0).min() <= 0 and not fill:
            raise ValueError("Masks contain non-filled areas")

        self.mask = mask / mask.sum(dim=0, keepdim=True)

    def on_apply_hooks(self, model: ModelPatcher, transformer_options: dict[str]):
        if not self.conds_kv:
            attn_patches = model.model_options["transformer_options"].get("patches", {}).get("attn2_patch", [])
            has_negpip = any("negpip_attn" in i.__name__ for i in attn_patches)
            log.debug("AttentionCouple has_negpip=%s", has_negpip)

            self.conds_kv = (
                [(cond[:, 0::2], cond[:, 1::2]) for cond in self.conds]
                if has_negpip
                else [(cond, cond) for cond in self.conds]
            )

            self.num_tokens_k = [cond[0].shape[1] for cond in self.conds_kv]
            self.num_tokens_v = [cond[1].shape[1] for cond in self.conds_kv]

        return super().on_apply_hooks(model, transformer_options)

    def to(self, *args, **kwargs):
        self.conds = [c.to(*args, **kwargs) for c in self.conds]
        self.mask = self.mask.to(*args, **kwargs)
        self.conds_kv = [(c1.to(*args, **kwargs), c2.to(*args, **kwargs)) for c1, c2 in self.conds_kv]
        return self

    def attn2_patch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options):
        cond_or_uncond = extra_options["cond_or_uncond"]

        num_chunks = len(cond_or_uncond) # should always be 1
        bs = q.shape[0] // num_chunks
        debug("attn2", f"{q.shape=} {k.shape=} {v.shape=} {bs=} {num_chunks=}")
        q_chunks = q.chunk(num_chunks, dim=0)
        k_chunks = k.chunk(num_chunks, dim=0)
        v_chunks = v.chunk(num_chunks, dim=0)
        lcm_tokens_k = lcm_for_list(self.num_tokens_k + [k.shape[1]])
        lcm_tokens_v = lcm_for_list(self.num_tokens_v + [v.shape[1]])
        conds_k_tensor = torch.cat(
            [
                cond[0].repeat(bs, lcm_tokens_k // self.num_tokens_k[i], 1) * self.strengths[i]
                for i, cond in enumerate(self.conds_kv)
            ],
            dim=0,
        )
        conds_v_tensor = torch.cat(
            [
                cond[1].repeat(bs, lcm_tokens_v // self.num_tokens_v[i], 1) * self.strengths[i]
                for i, cond in enumerate(self.conds_kv)
            ],
            dim=0,
        )

        qs, ks, vs = [], [], []
        for i, cond_type in enumerate(cond_or_uncond):
            q_target = q_chunks[i]
            k_target = k_chunks[i].repeat(1, lcm_tokens_k // k.shape[1], 1)
            v_target = v_chunks[i].repeat(1, lcm_tokens_v // v.shape[1], 1)
            qs.append(q_target.repeat(self.num_conds, 1, 1))
            ks.append(torch.cat([k_target * self.base_strength, conds_k_tensor], dim=0))
            vs.append(torch.cat([v_target * self.base_strength, conds_v_tensor], dim=0))

        qs = torch.cat(qs, dim=0)
        ks = torch.cat(ks, dim=0)
        vs = torch.cat(vs, dim=0)

        return qs, ks, vs

    def attn2_output_patch(self, out, extra_options):
        # out has been extended to shape [num_conds*batch_size, TOKENS, N]
        # out is [b1c1 b1c2 ... b1cN, b2c1 b2c2 ... b2cn, ...]
        num_conds = self.mask.shape[0]
        bs = out.shape[0] // num_conds
        num_tokens = out.shape[1]
        mask_size = extra_options["activations_shape"][-2:]
        mask_downsample = F.interpolate(self.mask, size=mask_size, mode="nearest")
        mask_downsample = mask_downsample.view(num_conds, num_tokens, 1).repeat_interleave(bs, dim=0)

        # cond_outputs is [num_conds*bs, tokens, N], output needs to be [bs, tokens, N]
        cond_outputs = out * mask_downsample
        cond_output = cond_outputs.view(num_conds, bs, out.shape[1], out.shape[2]).sum(0)
        return cond_output
