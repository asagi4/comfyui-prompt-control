# Lifted from https://github.com/pamparamm/ComfyUI-ppm/blob/c3e6b673ee2d424405dcb99aeed89f21943c89ac/nodes_ppm/attention_couple_ppm.py
# Original implementation by laksjdjf, hako-mikan, Haoming02 licensed under GPL-3.0
# https://github.com/laksjdjf/cgem156-ComfyUI/blob/1f5533f7f31345bafe4b833cbee15a3c4ad74167/scripts/attention_couple/node.py
# https://github.com/Haoming02/sd-forge-couple/blob/e8e258e982a8d149ba59a4bc43b945467604311c/scripts/attention_couple.py
import itertools
import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

from comfy.hooks import EnumHookScope, HookGroup, TransformerOptionsHook, set_hooks_for_conditioning
from comfy.model_patcher import ModelPatcher

log = logging.getLogger("comfyui-prompt-control")


def set_cond_attnmask(base_cond, extra_conds, fill=False):
    hook = AttentionCoupleHook()
    c = [base_cond[0][0], base_cond[0][1].copy()]
    # hook uses these, remove them to avoid doing latent masking
    c[1].pop("mask", None)
    c[1].pop("strength", None)
    c[1].pop("mask_strength", None)
    c = [c]
    c.extend(base_cond[1:])

    hook.initialize_regions(base_cond[0], extra_conds, fill=fill)
    group = HookGroup()
    group.add(hook)

    return set_hooks_for_conditioning(c, hooks=group, append_hooks=True)


def get_mask(mask, batch_size, num_tokens, extra_options):
    activations_shape = extra_options["activations_shape"]
    size = activations_shape[-2:]

    num_conds = mask.shape[0]
    mask_downsample = F.interpolate(mask, size=size, mode="nearest")
    mask_downsample_reshaped = mask_downsample.view(num_conds, num_tokens, 1).repeat_interleave(batch_size, dim=0)

    return mask_downsample_reshaped


class Proxy:
    def __init__(self, function):
        self.function = function

    def to(self, *args, **kwargs):
        self.function.__self__.to(*args, **kwargs)
        return self

    def __call__(self, *args, **kwargs):
        return self.function(*args, *kwargs)


class AttentionCoupleHook(TransformerOptionsHook):
    COND_UNCOND_COUPLE_OPTION = "cond_or_uncond_hook_couple"
    COND = 0
    UNCOND = 1

    def __init__(self):
        super().__init__(hook_scope=EnumHookScope.HookedOnly)

        self.transformers_dict = {
            "patches": {
                "attn2_output_patch": [Proxy(self.attn2_output_patch)],
                "attn2_patch": [Proxy(self.attn2_patch)],
            }
        }
        self.has_negpip = False

        # calculate later. All clones must refer to the same kv dict
        self.kv = {"k": None, "v": None}

    def initialize_regions(self, base_cond, conds, fill):
        self.num_conds = len(conds) + 1
        self.base_strength = base_cond[1].get("strength", 1.0)
        self.strengths = [cond[1].get("strength", 1.0) for cond in conds]
        self.conds: list[torch.Tensor] = [base_cond[0]] + [cond[0] for cond in conds]
        base_mask = base_cond[1].get("mask", None)
        masks = [cond[1].get("mask") * cond[1].get("mask_strength") for cond in conds]
        if len(masks) < 1:
            raise ValueError("Attention Couple hook makes no sense without masked conds")

        if any(m is None for m in masks):
            raise ValueError("All conds given to Attention Couple must have masks")

        if any(m.shape != masks[0].shape for m in masks) or (
            base_mask is not None and base_mask.shape != masks[0].shape
        ):
            largest_shape = max(m.shape for m in masks)
            if base_mask is not None:
                largest_shape = max(largest_shape, base_mask.shape)
            log.warning("Attention Couple: Masks are irregularly shaped, resizing them all to match the largest")
            for i in range(len(masks)):
                masks[i] = F.interpolate(masks[i].unsqueeze(1), size=largest_shape[1:], mode="nearest-exact").squeeze(1)

            if base_mask is not None:
                base_mask = F.interpolate(base_mask.unsqueeze(1), size=largest_shape[1:], mode="nearest-exact").squeeze(
                    1
                )

        if base_mask is None:
            if not fill:
                raise ValueError("You must specify a base mask when fill=False")
            sum = torch.stack(masks, dim=0).sum(dim=0)
            base_mask = torch.zeros_like(sum)
            base_mask[sum <= 0] = 1.0

        mask = [base_mask] + masks
        mask = torch.stack(mask, dim=0)
        if mask.sum(dim=0).min() <= 0 and not fill:
            raise ValueError("Masks contain non-filled areas")

        self.mask = mask / mask.sum(dim=0, keepdim=True)

    def on_apply_hooks(self, model: ModelPatcher, transformer_options: dict[str, Any]):
        if self.kv["k"] is None:
            self.has_negpip = model.model_options.get("ppm_negpip", False)
            log.debug("AttentionCouple has_negpip=%s", self.has_negpip)

            # Skip the base cond here, which is always first
            if self.has_negpip:
                self.kv["k"] = [cond[:, 0::2] for cond in self.conds[1:]]
                self.kv["v"] = [cond[:, 1::2] for cond in self.conds[1:]]
            else:
                self.kv["k"] = self.kv["v"] = self.conds[1:]

        return super().on_apply_hooks(model, transformer_options)

    def clone(self):
        c: AttentionCoupleHook = super().clone()
        c.mask = self.mask
        c.conds = self.conds
        c.kv = self.kv
        c.has_negpip = self.has_negpip
        c.base_strength = self.base_strength
        c.strengths = self.strengths
        c.num_conds = self.num_conds
        return c

    def to(self, *args, **kwargs):
        self.conds = [c.to(*args, **kwargs) for c in self.conds]
        self.mask = self.mask.to(*args, **kwargs)
        if self.kv["k"] is not None:
            self.kv["k"] = [c.to(*args, **kwargs) for c in self.kv["k"]]
            self.kv["v"] = [c.to(*args, **kwargs) for c in self.kv["v"]]
        return self

    def attn2_patch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options):
        cond_or_uncond = extra_options["cond_or_uncond"]
        cond_or_uncond_couple = extra_options[self.COND_UNCOND_COUPLE_OPTION] = list(cond_or_uncond)
        num_chunks = len(cond_or_uncond)

        # Cloning messes up the device sometimes
        if self.kv["k"][0].device != k.device:
            self.to(k)

        conds_k = self.kv["k"]
        conds_v = self.kv["v"]

        lcm_tokens_k = math.lcm(k.shape[1], *(cond.shape[1] for cond in conds_k))
        lcm_tokens_v = math.lcm(v.shape[1], *(cond.shape[1] for cond in conds_v))
        q_chunks = q.chunk(num_chunks, dim=0)
        k_chunks = k.chunk(num_chunks, dim=0)
        v_chunks = v.chunk(num_chunks, dim=0)

        bs = q.shape[0] // num_chunks

        conds_k_tensor = conds_v_tensor = torch.cat(
            [cond.repeat(bs, lcm_tokens_k // cond.shape[1], 1) * self.strengths[i] for i, cond in enumerate(conds_k)],
            dim=0,
        )
        if self.has_negpip:
            conds_v_tensor = torch.cat(
                [
                    cond.repeat(bs, lcm_tokens_v // cond.shape[1], 1) * self.strengths[i]
                    for i, cond in enumerate(conds_v)
                ],
                dim=0,
            )

        qs, ks, vs = [], [], []
        cond_or_uncond_couple.clear()

        for i, cond_type in enumerate(cond_or_uncond):
            q_target = q_chunks[i]
            k_target = k_chunks[i].repeat(1, lcm_tokens_k // k.shape[1], 1)
            v_target = v_chunks[i].repeat(1, lcm_tokens_v // v.shape[1], 1)
            if cond_type == self.UNCOND:
                qs.append(q_target)
                ks.append(k_target)
                vs.append(v_target)
                cond_or_uncond_couple.append(self.UNCOND)
            else:
                qs.append(q_target.repeat(self.num_conds, 1, 1))
                ks.append(
                    torch.cat(
                        [
                            k_target * self.base_strength,
                            conds_k_tensor,
                        ],
                        dim=0,
                    )
                )
                vs.append(
                    torch.cat(
                        [
                            v_target * self.base_strength,
                            conds_v_tensor,
                        ],
                        dim=0,
                    )
                )
                cond_or_uncond_couple.extend(itertools.repeat(self.COND, self.num_conds))

        q = torch.cat(qs, dim=0)
        k = torch.cat(ks, dim=0)
        v = torch.cat(vs, dim=0)

        return q, k, v

    def attn2_output_patch(self, out, extra_options):
        cond_or_uncond = extra_options[self.COND_UNCOND_COUPLE_OPTION]
        bs = out.shape[0] // len(cond_or_uncond)
        mask_downsample = get_mask(self.mask, bs, out.shape[1], extra_options)
        outputs = []
        cond_outputs = []
        i_cond = 0
        for i, cond_type in enumerate(cond_or_uncond):
            pos, next_pos = i * bs, (i + 1) * bs

            if cond_type == self.UNCOND:
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
