import logging

log = logging.getLogger("comfyui-prompt-control")

from comfy.hooks import TransformerOptionsHook, HookGroup, EnumHookScope
from comfy.ldm.modules.attention import optimized_attention
import torch.nn.functional as F
import torch
from math import sqrt, gcd


def get_mask(mask, batch_size, num_tokens, original_shape):
    num_conds = mask.shape[0]

    if original_shape[2] * original_shape[3] == num_tokens:
        down_sample_rate = 1
    elif (original_shape[2] // 2) * (original_shape[3] // 2) == num_tokens:
        down_sample_rate = 2
    elif (original_shape[2] // 4) * (original_shape[3] // 4) == num_tokens:
        down_sample_rate = 4
    else:
        down_sample_rate = 8

    size = (original_shape[2] // down_sample_rate, original_shape[3] // down_sample_rate)
    mask_downsample = F.interpolate(mask, size=size, mode="nearest")
    mask_downsample = mask_downsample.view(num_conds, num_tokens, 1).repeat_interleave(batch_size, dim=0)

    return mask_downsample


def lcm(a, b):
    return a * b // gcd(a, b)


def lcm_for_list(numbers):
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = lcm(current_lcm, number)
    return current_lcm


def attention_couple_simple(base_mask, conds, masks):
    num_conds = len(conds) + 1
    mask = [base_mask] + masks
    mask = torch.stack(mask, dim=0)
    assert mask.sum(dim=0).min() > 0, "There are areas that are zero in all masks."
    self_mask = mask / mask.sum(dim=0, keepdim=True)
    self_conds = [cond[0][0] for cond in conds]
    num_tokens = [cond.shape[1] for cond in self_conds]
    self_batch_size = None

    def attn2_patch(q, k, v, extra_options):
        nonlocal self_conds
        nonlocal self_mask
        nonlocal self_batch_size
        assert k.mean() == v.mean(), "k and v must be the same."
        device, dtype = q.device, q.dtype

        if self_conds[0].device != device:
            self_conds = [cond.to(device, dtype=dtype) for cond in self_conds]
        if self_mask.device != device:
            self_mask = self_mask.to(device, dtype=dtype)

        cond_or_unconds = extra_options["cond_or_uncond"]
        num_chunks = len(cond_or_unconds)
        self_batch_size = q.shape[0] // num_chunks
        q_chunks = q.chunk(num_chunks, dim=0)
        k_chunks = k.chunk(num_chunks, dim=0)
        lcm_tokens = lcm_for_list(num_tokens + [k.shape[1]])
        conds_tensor = torch.cat(
            [cond.repeat(self_batch_size, lcm_tokens // num_tokens[i], 1) for i, cond in enumerate(self_conds)], dim=0
        )

        qs, ks = [], []
        for i, cond_or_uncond in enumerate(cond_or_unconds):
            k_target = k_chunks[i].repeat(1, lcm_tokens // k.shape[1], 1)
            if cond_or_uncond == 1:  # uncond
                qs.append(q_chunks[i])
                ks.append(k_target)
            else:
                qs.append(q_chunks[i].repeat(num_conds, 1, 1))
                ks.append(torch.cat([k_target, conds_tensor], dim=0))

        qs = torch.cat(qs, dim=0)
        ks = torch.cat(ks, dim=0).to(k)

        return qs, ks, ks

    def attn2_output_patch(out, extra_options):
        nonlocal self_conds
        nonlocal self_mask
        nonlocal self_batch_size
        cond_or_unconds = extra_options["cond_or_uncond"]
        mask_downsample = get_mask(self_mask, self_batch_size, out.shape[1], extra_options["original_shape"])
        outputs = []
        pos = 0
        for cond_or_uncond in cond_or_unconds:
            if cond_or_uncond == 1:  # uncond
                outputs.append(out[pos : pos + self_batch_size])
                pos += self_batch_size
            else:
                masked_output = (out[pos : pos + num_conds * self_batch_size] * mask_downsample).view(
                    num_conds, self_batch_size, out.shape[1], out.shape[2]
                )
                masked_output = masked_output.sum(dim=0)
                outputs.append(masked_output)
                pos += num_conds * self_batch_size
        return torch.cat(outputs, dim=0)

    transformers_dict = {"patches": {"attn2_output_patch": [attn2_output_patch], "attn2_patch": [attn2_patch]}}
    hook = TransformerOptionsHook(transformers_dict=transformers_dict, hook_scope=EnumHookScope.HookedOnly)
    group = HookGroup()
    group.add(hook)
    return group


class MaskedAttn2:
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, q, k, v, extra_options):
        mask = self.mask
        orig_shape = extra_options["original_shape"]
        _, _, oh, ow = orig_shape
        seq_len = q.shape[1]
        mask_h = oh / sqrt(oh * ow / seq_len)
        mask_h = int(mask_h) + int((seq_len % int(mask_h)) != 0)
        mask_w = seq_len // mask_h
        r = optimized_attention(q, k, v, extra_options["n_heads"])
        mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="nearest").squeeze(1)
        mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, r.shape[2])

        return mask * r


def create_attention_hook(mask):
    attn_replacements = {}
    mask = mask.detach().to(device="cuda", dtype=torch.float16)

    masked_attention = MaskedAttn2(mask)

    for id in [4, 5, 7, 8]:  # id of input_blocks that have cross attention
        block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
        for index in block_indices:
            k = ("input", id, index)
            attn_replacements[k] = masked_attention
    for id in range(6):  # id of output_blocks that have cross attention
        block_indices = range(2) if id in [3, 4, 5] else range(10)  # transformer_depth
        for index in block_indices:
            k = ("output", id, index)
            attn_replacements[k] = masked_attention
    for index in range(10):
        k = ("middle", 1, index)
        attn_replacements[k] = masked_attention

    hook = TransformerOptionsHook(
        transformers_dict={"patches_replace": {"attn2": attn_replacements}}, hook_scope=EnumHookScope.HookedOnly
    )
    group = HookGroup()
    group.add(hook)

    return group


class AttentionMaskHookExperimental:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("HOOKS",)
    CATEGORY = "promptcontrol/_testing"
    FUNCTION = "apply"
    EXPERIMENTAL = True
    DESCRIPTION = "Experimental attention masking hook. For testing only"

    def apply(self, mask):
        return (create_attention_hook(mask),)


NODE_CLASS_MAPPINGS = {"AttentionMaskHookExperimental": AttentionMaskHookExperimental}

NODE_DISPLAY_NAME_MAPPINGS = {}
