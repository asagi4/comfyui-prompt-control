# Lifted from https://github.com/pamparamm/ComfyUI-ppm/blob/c3e6b673ee2d424405dcb99aeed89f21943c89ac/nodes_ppm/attention_couple_ppm.py
# Original implementation by laksjdjf, hako-mikan, Haoming02 licensed under GPL-3.0
# https://github.com/laksjdjf/cgem156-ComfyUI/blob/1f5533f7f31345bafe4b833cbee15a3c4ad74167/scripts/attention_couple/node.py
# https://github.com/Haoming02/sd-forge-couple/blob/e8e258e982a8d149ba59a4bc43b945467604311c/scripts/attention_couple.py
import itertools
import math

import torch
import torch.nn.functional as F
from comfy.hooks import TransformerOptionsHook, HookGroup, EnumHookScope, set_hooks_for_conditioning

from comfy.model_patcher import ModelPatcher

import logging

log = logging.getLogger("comfyui-prompt-control")

COND = 0
UNCOND = 1
COND_UNCOND_COUPLE = "cond_or_uncond_couple"


def set_cond_attnmask(base_cond, base_mask, conds, masks):
    hook = AttentionCoupleHook(base_mask, conds, masks)
    group = HookGroup()
    group.add(hook)
    return set_hooks_for_conditioning(base_cond, hooks=group)


def get_mask(mask, batch_size, num_tokens, extra_options):
    activations_shape = extra_options["activations_shape"]
    size = activations_shape[-2:]

    num_conds = mask.shape[0]
    mask_downsample = F.interpolate(mask, size=size, mode="nearest")
    mask_downsample_reshaped = mask_downsample.view(num_conds, num_tokens, 1).repeat_interleave(batch_size, dim=0)

    return mask_downsample_reshaped


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
    def __init__(self, base_mask, conds, masks):
        super().__init__(hook_scope=EnumHookScope.HookedOnly)
        self.transformers_dict = {
            "patches": {
                "attn2_output_patch": [Proxy(self.attn2_output_patch)],
                "attn2_patch": [Proxy(self.attn2_patch)],
            }
        }

        self.conds_kv = []

        self.batch_size = 0
        self.num_conds = len(conds) + 1

        mask = [base_mask] + masks
        mask = torch.stack(mask, dim=0)
        if mask.sum(dim=0).min() <= 0:
            raise ValueError("Masks contain non-filled areas")
        self.mask = mask / mask.sum(dim=0, keepdim=True)

        self.conds: list[torch.Tensor] = [cond[0][0] for cond in conds]

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

        num_chunks = len(cond_or_uncond)
        self.batch_size = q.shape[0] // num_chunks
        if len(self.conds_kv) > 0:
            q_chunks = q.chunk(num_chunks, dim=0)
            k_chunks = k.chunk(num_chunks, dim=0)
            v_chunks = v.chunk(num_chunks, dim=0)
            lcm_tokens_k = lcm_for_list(self.num_tokens_k + [k.shape[1]])
            lcm_tokens_v = lcm_for_list(self.num_tokens_v + [v.shape[1]])
            conds_k_tensor = torch.cat(
                [
                    cond[0].repeat(self.batch_size, lcm_tokens_k // self.num_tokens_k[i], 1)
                    for i, cond in enumerate(self.conds_kv)
                ],
                dim=0,
            )
            conds_v_tensor = torch.cat(
                [
                    cond[1].repeat(self.batch_size, lcm_tokens_v // self.num_tokens_v[i], 1)
                    for i, cond in enumerate(self.conds_kv)
                ],
                dim=0,
            )

            qs, ks, vs = [], [], []
            cond_or_uncond_couple = []
            for i, cond_type in enumerate(cond_or_uncond):
                q_target = q_chunks[i]
                k_target = k_chunks[i].repeat(1, lcm_tokens_k // k.shape[1], 1)
                v_target = v_chunks[i].repeat(1, lcm_tokens_v // v.shape[1], 1)
                if cond_type == UNCOND:
                    qs.append(q_target)
                    ks.append(k_target)
                    vs.append(v_target)
                    cond_or_uncond_couple.append(UNCOND)
                else:
                    qs.append(q_target.repeat(self.num_conds, 1, 1))
                    ks.append(torch.cat([k_target, conds_k_tensor], dim=0))
                    vs.append(torch.cat([v_target, conds_v_tensor], dim=0))
                    cond_or_uncond_couple.extend(itertools.repeat(COND, self.num_conds))

            qs = torch.cat(qs, dim=0)
            ks = torch.cat(ks, dim=0)
            vs = torch.cat(vs, dim=0)

            extra_options[COND_UNCOND_COUPLE] = cond_or_uncond_couple

            return qs, ks, vs

        return q, k, v

    def attn2_output_patch(self, out, extra_options):
        cond_or_uncond = extra_options[COND_UNCOND_COUPLE]
        bs = self.batch_size
        mask_downsample = get_mask(self.mask, self.batch_size, out.shape[1], extra_options)
        outputs = []
        cond_outputs = []
        i_cond = 0
        for i, cond_type in enumerate(cond_or_uncond):
            pos, next_pos = i * bs, (i + 1) * bs

            if cond_type == UNCOND:
                outputs.append(out[pos:next_pos])
            else:
                pos_cond, next_pos_cond = i_cond * bs, (i_cond + 1) * bs
                masked_output = out[pos:next_pos] * mask_downsample[pos_cond:next_pos_cond]
                cond_outputs.append(masked_output)
                i_cond += 1

        cond_output = torch.stack(cond_outputs).sum(0)
        outputs.append(cond_output)
        return torch.cat(outputs, dim=0)
