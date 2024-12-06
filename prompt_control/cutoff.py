import torch
import copy
import re
import warnings

import numpy as np

from .adv_encode import (
    advanced_encode_from_tokens,
    encode_token_weights_g,
    encode_token_weights_l,
    encode_token_weights,
    prepareXL,
)
from comfy.sdxl_clip import SDXLClipModel, SDXLRefinerClipModel, SDXLClipG

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


def replace_embeddings(max_token, prompt, replacements=None):

    if replacements is None:
        emb_lookup = []
    else:
        emb_lookup = replacements.copy()
        max_token += len(emb_lookup)

    def get_replacement(embedding):
        for e, n in emb_lookup:
            if torch.equal(embedding, e):
                return n
        return None

    tokens = []
    for x in prompt:
        row = []
        for i in range(len(x)):
            emb = x[i][0]
            if not torch.is_tensor(emb):
                row.append(emb)
            else:
                n = get_replacement(emb)
                if n is not None:
                    row.append(n)
                else:
                    max_token += 1
                    row.append(max_token)
                    emb_lookup.append((emb, max_token))
        tokens.append(row)
    tokens = np.array(tokens)[:, 1:-1].reshape(-1)
    return (tokens, emb_lookup)


def unpad_prompt(end_token, prompt):
    res = np.trim_zeros(prompt, "b")
    return np.trim_zeros(res - end_token, "b") + end_token


class CLIPRegionsBasePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}), "clip": ("CLIP",)}}

    RETURN_TYPES = ("CLIPREGION",)
    FUNCTION = "init_prompt"

    CATEGORY = "conditioning/cutoff"

    def init_prompt(self, clip, text):
        tokens = clip.tokenize(text, return_word_ids=True)
        return (
            {
                "clip": clip,
                "base_tokens": tokens,
                "regions": [],
                "targets": [],
                "weights": [],
            },
        )


def get_sublists(super_list, sub_list):
    positions = []
    for candidate_ind in (i for i, e in enumerate(super_list) if e == sub_list[0]):
        if super_list[candidate_ind : candidate_ind + len(sub_list)] == sub_list:
            positions.append(candidate_ind)
    return positions


class CLIPSetRegion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_regions": ("CLIPREGION",),
                "region_text": ("STRING", {"multiline": True}),
                "target_text": ("STRING", {"multiline": False}),
                "weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("CLIPREGION",)
    FUNCTION = "add_clip_region"

    CATEGORY = "conditioning/cutoff"

    def add_clip_region(self, clip_regions, region_text, target_text, weight):
        clip = clip_regions["clip"]
        tokenizer = clip.tokenizer
        base_tokens = clip_regions["base_tokens"]
        if "g" in base_tokens:
            base_tokens = base_tokens["g"]
        elif "l" in base_tokens:
            base_tokens = base_tokens["l"]
        else:
            raise Exception("No recognized tokenizer")
        if hasattr(tokenizer, "clip_g"):
            tokenizer = tokenizer.clip_g
        elif hasattr(tokenizer, "clip_l"):
            tokenizer = tokenizer.clip_l
        else:
            raise Exception("No recognized tokenizer")
        region_outputs = []
        target_outputs = []

        # strip input strings
        region_text = region_text.strip()
        target_text = target_text.strip()

        endtoken = tokenizer.end_token

        prompt_tokens, emb_lookup = replace_embeddings(endtoken, base_tokens)

        for rt in region_text.split("\n"):
            region_tokens = tokenizer.tokenize_with_weights(rt)
            region_tokens, _ = replace_embeddings(endtoken, region_tokens, emb_lookup)
            region_tokens = unpad_prompt(endtoken, region_tokens).tolist()

            # calc region mask
            region_length = len(region_tokens)
            regions = get_sublists(list(prompt_tokens), region_tokens)

            region_mask = np.zeros(len(prompt_tokens))
            for r in regions:
                region_mask[r : r + region_length] = 1
            region_mask = region_mask.reshape(-1, tokenizer.max_length - 2)
            region_mask = np.pad(region_mask, pad_width=((0, 0), (1, 1)), mode="constant", constant_values=0)
            region_mask = region_mask.reshape(1, -1)
            region_outputs.append(region_mask)

            # calc target mask
            targets = []
            for target in target_text.split(" "):
                # deal with underscores
                target = re.sub(r"(?<!\\)_", " ", target)
                target = re.sub(r"\\_", "_", target)

                target_tokens = tokenizer.tokenize_with_weights(target)
                target_tokens, _ = replace_embeddings(endtoken, target_tokens, emb_lookup)
                target_tokens = unpad_prompt(endtoken, target_tokens).tolist()

                targets.extend([(x, len(target_tokens)) for x in get_sublists(region_tokens, target_tokens)])
            targets = [(t_start + r, t_start + t_end + r) for r in regions for t_start, t_end in targets]

            targets_mask = np.zeros(len(prompt_tokens))
            for t_start, t_end in targets:
                targets_mask[t_start:t_end] = 1
            targets_mask = targets_mask.reshape(-1, tokenizer.max_length - 2)
            targets_mask = np.pad(targets_mask, pad_width=((0, 0), (1, 1)), mode="constant", constant_values=0)
            targets_mask = targets_mask.reshape(1, -1)
            target_outputs.append(targets_mask)

        # prepare output
        region_mask_list = clip_regions["regions"].copy()
        region_mask_list.extend(region_outputs)
        target_mask_list = clip_regions["targets"].copy()
        target_mask_list.extend(target_outputs)
        weight_list = clip_regions["weights"].copy()
        weight_list.extend([weight] * len(region_outputs))

        return (
            {
                "clip": clip,
                "base_tokens": clip_regions["base_tokens"],
                "regions": region_mask_list,
                "targets": target_mask_list,
                "weights": weight_list,
            },
        )


def create_masked_prompt(weighted_tokens, mask, mask_token):
    if isinstance(weighted_tokens, dict):
        result = dict()
        for k in weighted_tokens.keys():
            result[k] = _create_masked_prompt(weighted_tokens[k], mask, mask_token)
        return result
    else:
        return _create_masked_prompt(weighted_tokens, mask, mask_token)


def _create_masked_prompt(weighted_tokens, mask, mask_token):
    mask_ids = list(zip(*np.nonzero(mask.reshape((len(weighted_tokens), -1)))))
    new_prompt = copy.deepcopy(weighted_tokens)
    for x, y in mask_ids:
        new_prompt[x][y] = (mask_token,) + new_prompt[x][y][1:]
    return new_prompt


def encode_from_tokens(clip, tokenized, token_normalization, weight_interpretation, return_pooled=False):
    if isinstance(clip.cond_stage_model, (SDXLClipModel, SDXLRefinerClipModel, SDXLClipG)):
        embs_l = None
        embs_g = None
        pooled = None
        if "l" in tokenized and isinstance(clip.cond_stage_model, SDXLClipModel):
            embs_l, _ = advanced_encode_from_tokens(
                tokenized["l"],
                token_normalization,
                weight_interpretation,
                lambda x: encode_token_weights(clip, x, encode_token_weights_l),
                w_max=1.0,
                return_pooled=False,
            )
        if "g" in tokenized:
            embs_g, pooled = advanced_encode_from_tokens(
                tokenized["g"],
                token_normalization,
                weight_interpretation,
                lambda x: encode_token_weights(clip, x, encode_token_weights_g),
                w_max=1.0,
                return_pooled=True,
            )
        emb, pool = prepareXL(embs_l, embs_g, pooled, 0.5)
    else:
        emb, pool = advanced_encode_from_tokens(
            tokenized["l"],
            token_normalization,
            weight_interpretation,
            lambda x: (clip.encode_from_tokens({"l": x}), None),
            w_max=1.0,
        )
    if return_pooled:
        return emb, pool
    else:
        return emb


def finalize_clip_regions(
    clip_regions, mask_token, strict_mask, start_from_masked, token_normalization="none", weight_interpretation="comfy"
):
    clip = clip_regions["clip"]
    tokenizer = clip.tokenizer
    if hasattr(tokenizer, "clip_g"):
        tokenizer = tokenizer.clip_g
    base_weighted_tokens = clip_regions["base_tokens"]

    # calc base embedding
    base_embedding_full, pool = encode_from_tokens(
        clip, base_weighted_tokens, token_normalization, weight_interpretation, True
    )

    # Avoid numpy value error and passthrough base embeddings if no regions are set.
    if len(clip_regions["regions"]) == 0:
        return ([[base_embedding_full, {"pooled_output": pool}]],)

    if mask_token == "":
        mask_token = 266  # clip.tokenizer.end_token
    else:
        mask_token = tokenizer.tokenizer(mask_token)["input_ids"][1:-1]
        if len(mask_token) > 1:
            warnings.warn("mask_token does not map to a single token, using the first token instead")
        mask_token = mask_token[0]

    # calc global target mask
    global_target_mask = np.any(np.stack(clip_regions["targets"]), axis=0).astype(int)

    # calc global region mask
    global_region_mask = np.any(np.stack(clip_regions["regions"]), axis=0).astype(float)
    regions_sum = np.sum(np.stack(clip_regions["regions"]), axis=0)
    regions_normalized = np.divide(1, regions_sum, out=np.zeros_like(regions_sum), where=regions_sum != 0)

    # mask base embeddings
    base_embedding_masked = encode_from_tokens(
        clip,
        create_masked_prompt(base_weighted_tokens, global_target_mask, mask_token),
        token_normalization,
        weight_interpretation,
    )
    base_embedding_start = base_embedding_full * (1 - start_from_masked) + base_embedding_masked * start_from_masked
    base_embedding_outer = base_embedding_full * (1 - strict_mask) + base_embedding_masked * strict_mask

    region_embeddings = []
    for region, target, weight in zip(clip_regions["regions"], clip_regions["targets"], clip_regions["weights"]):
        region_masking = torch.tensor(
            regions_normalized * region * weight, dtype=base_embedding_full.dtype, device=base_embedding_full.device
        ).unsqueeze(-1)

        region_emb = encode_from_tokens(
            clip,
            create_masked_prompt(base_weighted_tokens, global_target_mask - target, mask_token),
            token_normalization,
            weight_interpretation,
        )
        region_emb -= base_embedding_start
        region_emb *= region_masking

        region_embeddings.append(region_emb)
    region_embeddings = torch.stack(region_embeddings).sum(axis=0)

    embeddings_final_mask = torch.tensor(
        global_region_mask, dtype=base_embedding_full.dtype, device=base_embedding_full.device
    ).unsqueeze(-1)
    embeddings_final = base_embedding_start * embeddings_final_mask + base_embedding_outer * (1 - embeddings_final_mask)
    embeddings_final += region_embeddings
    return ([[embeddings_final, {"pooled_output": pool}]],)


class CLIPRegionsToConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_regions": ("CLIPREGION",),
                "mask_token": ("STRING", {"multiline": False, "default": ""}),
                "strict_mask": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "start_from_masked": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "finalize"

    CATEGORY = "conditioning/cutoff"

    def finalize(self, clip_regions, mask_token, strict_mask, start_from_masked):
        return finalize_clip_regions(clip_regions, mask_token, strict_mask, start_from_masked)


class CLIPRegionsToConditioningADV:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_regions": ("CLIPREGION",),
                "mask_token": ("STRING", {"multiline": False, "default": ""}),
                "strict_mask": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "start_from_masked": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "finalize"

    CATEGORY = "conditioning/cutoff"

    def finalize(
        self, clip_regions, mask_token, strict_mask, start_from_masked, token_normalization, weight_interpretation
    ):
        return finalize_clip_regions(
            clip_regions, mask_token, strict_mask, start_from_masked, token_normalization, weight_interpretation
        )


NODE_CLASS_MAPPINGS = {
    "BNK_CutoffBasePrompt": CLIPRegionsBasePrompt,
    "BNK_CutoffSetRegions": CLIPSetRegion,
    "BNK_CutoffRegionsToConditioning": CLIPRegionsToConditioning,
    "BNK_CutoffRegionsToConditioning_ADV": CLIPRegionsToConditioningADV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BNK_CutoffBasePrompt": "Cutoff Base Prompt",
    "BNK_CutoffSetRegions": "Cutoff Set Regions",
    "BNK_CutoffRegionsToConditioning": "Cutoff Regions To Conditioning",
    "BNK_CutoffRegionsToConditioning_ADV": "Cutoff Regions To Conditioning (ADV)",
}
