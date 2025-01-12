import logging

log = logging.getLogger("comfyui-prompt-control")

from comfy.hooks import TransformerOptionsHook, HookGroup, EnumHookScope
from comfy.ldm.modules.attention import optimized_attention
import torch.nn.functional as F
import torch
from math import sqrt


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
