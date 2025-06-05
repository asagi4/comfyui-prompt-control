import torch
import copy
import re

import numpy as np
import logging

log = logging.getLogger("comfyui-prompt-control")


def replace_embeddings(max_token, prompt, replacements=None):
    """Replaces embedding tensors in a token array and replaces them with increasing IDs past max_token"""

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


def unpad_prompt(pad_token, prompt):
    res = np.trim_zeros(prompt, "b")
    return np.trim_zeros(res - pad_token, "b") + pad_token


def get_sublists(super_list, sub_list):
    positions = []
    for candidate_ind in (i for i, e in enumerate(super_list) if e == sub_list[0]):
        if super_list[candidate_ind : candidate_ind + len(sub_list)] == sub_list:
            positions.append(candidate_ind)
    return positions


def cutoff_add_region(
    clip_regions, tokenizer, region_text, target_text, weight, strict_mask, start_from_masked, mask_token
):
    """Adds a cut region to the clip_regions dictionary. It is modified in place"""
    base_tokens = clip_regions["base_tokens"]
    region_outputs = []
    target_outputs = []
    if strict_mask is not None:
        clip_regions["strict_mask"] = float(strict_mask)
    if start_from_masked is not None:
        clip_regions["start_from_masked"] = float(start_from_masked)
    if mask_token is not None:
        clip_regions["mask_token"] = tokenizer.tokenizer(mask_token)["input_ids"][1]
    if weight is None:
        weight = 1.0
    else:
        weight = float(weight)

    region_text = region_text.strip()
    target_text = target_text.strip()

    strict_mask = clip_regions["strict_mask"]
    start_from_masked = clip_regions["start_from_masked"]
    mask_token = clip_regions["mask_token"]
    log.info(f"CUT region {region_text=} {target_text=} {weight=} {strict_mask=} {start_from_masked=} {mask_token=}")

    pad_token = tokenizer.end_token

    prompt_tokens, emb_lookup = replace_embeddings(pad_token, base_tokens)

    for rt in region_text.split("\n"):
        region_tokens = tokenizer.tokenize_with_weights(rt)
        region_tokens, _ = replace_embeddings(pad_token, region_tokens, emb_lookup)
        region_tokens = unpad_prompt(pad_token, region_tokens).tolist()

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
            target_tokens, _ = replace_embeddings(pad_token, target_tokens, emb_lookup)
            target_tokens = unpad_prompt(pad_token, target_tokens).tolist()

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

    clip_regions["regions"] = region_mask_list
    clip_regions["targets"] = target_mask_list
    clip_regions["weights"] = weight_list


def create_masked_prompt(weighted_tokens, mask, mask_token):
    mask_ids = list(zip(*np.nonzero(mask.reshape((len(weighted_tokens), -1)))))
    new_prompt = copy.deepcopy(weighted_tokens)
    for x, y in mask_ids:
        new_prompt[x][y] = (mask_token,) + new_prompt[x][y][1:]
    return new_prompt


def process_cuts(encode, extra, tokens):
    if not extra.get("cuts"):
        return encode(tokens)

    base = {
        "base_tokens": tokens,
        "regions": [],
        "targets": [],
        "weights": [],
        "strict_mask": 1.0,
        "start_from_masked": 1.0,
        "mask_token": extra["tokenizer"].tokenizer("+")["input_ids"][1],
    }

    for cut in extra["cuts"]:
        cutoff_add_region(base, extra["tokenizer"], *cut)

    return encode_regions(base, encode, extra["tokenizer"])


def debug_tokens(label, prompt, tokenizer):
    log.debug("Tokens for %s", label)
    for tokens in prompt:
        tokens = (t for t in tokens if not torch.is_tensor(t[0]))
        log.debug(" ".join(f"{x[0][0]} {x[1]}" for x in tokenizer.untokenize(tokens) if x[0][0] != tokenizer.end_token))


def encode_regions(clip_regions, encode, tokenizer):
    base_weighted_tokens = clip_regions["base_tokens"]
    start_from_masked = clip_regions["start_from_masked"]
    mask_token = clip_regions["mask_token"]
    strict_mask = clip_regions["strict_mask"]

    # calc base embedding
    base_embedding_full, pool = encode(base_weighted_tokens)

    # Avoid numpy value error and passthrough base embeddings if no regions are set.

    # calc global target mask
    global_target_mask = np.any(np.stack(clip_regions["targets"]), axis=0).astype(int)

    # calc global region mask
    global_region_mask = np.any(np.stack(clip_regions["regions"]), axis=0).astype(float)
    regions_sum = np.sum(np.stack(clip_regions["regions"]), axis=0)
    regions_normalized = np.divide(1, regions_sum, out=np.zeros_like(regions_sum), where=regions_sum != 0)

    # mask base embeddings
    base_masked_prompt = create_masked_prompt(base_weighted_tokens, global_target_mask, mask_token)
    debug_tokens("base_masked", base_masked_prompt, tokenizer)
    base_embedding_masked, _ = encode(base_masked_prompt)
    base_embedding_start = base_embedding_full * (1 - start_from_masked) + base_embedding_masked * start_from_masked
    base_embedding_outer = base_embedding_full * (1 - strict_mask) + base_embedding_masked * strict_mask

    region_embeddings = []
    for region, target, weight in zip(clip_regions["regions"], clip_regions["targets"], clip_regions["weights"]):
        region_masking = torch.tensor(
            regions_normalized * region * weight, dtype=base_embedding_full.dtype, device=base_embedding_full.device
        ).unsqueeze(-1)

        region_prompt = create_masked_prompt(base_weighted_tokens, global_target_mask - target, mask_token)
        debug_tokens("region", region_prompt, tokenizer)
        region_emb, _ = encode(region_prompt)
        region_emb -= base_embedding_start
        # NegPiP support:
        if region_emb.shape[1] == 2 * region_masking.shape[1]:
            region_masking = torch.repeat_interleave(region_masking, 2, dim=1)
        region_emb *= region_masking

        region_embeddings.append(region_emb)
    region_embeddings = torch.stack(region_embeddings).sum(axis=0)

    embeddings_final_mask = torch.tensor(
        global_region_mask, dtype=base_embedding_full.dtype, device=base_embedding_full.device
    ).unsqueeze(-1)
    # NegPiP support:
    if region_embeddings.shape[1] == 2 * embeddings_final_mask.shape[1]:
        embeddings_final_mask = torch.repeat_interleave(embeddings_final_mask, 2, dim=1)

    embeddings_final = base_embedding_start * embeddings_final_mask + base_embedding_outer * (1 - embeddings_final_mask)
    embeddings_final += region_embeddings
    return embeddings_final, pool
