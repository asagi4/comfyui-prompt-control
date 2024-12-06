import torch
import numpy as np
import itertools


def _grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def _norm_mag(w, n):
    d = w - 1
    return 1 + np.sign(d) * np.sqrt(np.abs(d) ** 2 / n)
    # return  np.sign(w) * np.sqrt(np.abs(w)**2 / n)


def weights_like(weights, emb):
    return torch.tensor(weights, dtype=emb.dtype, device=emb.device).reshape(1, -1, 1).expand(emb.shape)


def divide_length(word_ids, weights):
    sums = dict(zip(*np.unique(word_ids, return_counts=True)))
    sums[0] = 1
    weights = [[_norm_mag(w, sums[id]) if id != 0 else 1.0 for w, id in zip(x, y)] for x, y in zip(weights, word_ids)]
    return weights


def shift_mean_weight(word_ids, weights):
    delta = 1 - np.mean([w for x, y in zip(weights, word_ids) for w, id in zip(x, y) if id != 0])
    weights = [[w if id == 0 else w + delta for w, id in zip(x, y)] for x, y in zip(weights, word_ids)]
    return weights


def scale_to_norm(weights, word_ids, w_max):
    top = np.max(weights)
    w_max = min(top, w_max)
    weights = [[w_max if id == 0 else (w / top) * w_max for w, id in zip(x, y)] for x, y in zip(weights, word_ids)]
    return weights


def mask_word_id(tokens, word_ids, target_id, mask_token):
    new_tokens = [[mask_token if wid == target_id else t for t, wid in zip(x, y)] for x, y in zip(tokens, word_ids)]
    mask = np.array(word_ids) == target_id
    return (new_tokens, mask)


def batched_clip_encode(tokens, length, encode_func, num_chunks):
    embs = []
    for e in _grouper(32, tokens):
        enc, pooled = encode_func(e)
        enc = enc.reshape((len(e), length, -1))
        embs.append(enc)
    embs = torch.cat(embs)
    embs = embs.reshape((len(tokens) // num_chunks, length * num_chunks, -1))
    return embs


def from_masked(tokens, weights, word_ids, base_emb, length, encode_func, m_token=266):
    pooled_base = base_emb[0, length - 1 : length, :]
    wids, inds = np.unique(np.array(word_ids).reshape(-1), return_index=True)
    weight_dict = dict((id, w) for id, w in zip(wids, np.array(weights).reshape(-1)[inds]) if w != 1.0)

    if len(weight_dict) == 0:
        return torch.zeros_like(base_emb), base_emb[0, length - 1 : length, :]

    weight_tensor = weights_like(weights, base_emb)

    # m_token = (clip.tokenizer.end_token, 1.0) if  clip.tokenizer.pad_with_end else (0,1.0)
    # TODO: find most suitable masking token here
    m_token = (m_token, 1.0)

    ws = []
    masked_tokens = []
    masks = []

    # create prompts
    for id, w in weight_dict.items():
        masked, m = mask_word_id(tokens, word_ids, id, m_token)
        masked_tokens.extend(masked)
        masks.append(weights_like(m, base_emb))

        ws.append(w)

    # batch process prompts
    embs = batched_clip_encode(masked_tokens, length, encode_func, len(tokens))
    masks = torch.cat(masks)

    embs = base_emb.expand(embs.shape) - embs
    pooled = embs[0, length - 1 : length, :]

    embs *= masks
    embs = embs.sum(axis=0, keepdim=True)

    pooled_start = pooled_base.expand(len(ws), -1)
    ws = torch.tensor(ws).reshape(-1, 1).expand(pooled_start.shape)
    pooled = (pooled - pooled_start) * (ws - 1)
    pooled = pooled.mean(axis=0, keepdim=True)

    return ((weight_tensor - 1) * embs), pooled_base + pooled


def mask_inds(tokens, inds, mask_token):
    clip_len = len(tokens[0])
    inds_set = set(inds)
    new_tokens = [
        [mask_token if i * clip_len + j in inds_set else t for j, t in enumerate(x)] for i, x in enumerate(tokens)
    ]
    return new_tokens


def down_weight(tokens, weights, word_ids, base_emb, length, encode_func, m_token=266):
    w, w_inv = np.unique(weights, return_inverse=True)

    if np.sum(w < 1) == 0:
        return base_emb, tokens, base_emb[0, length - 1 : length, :]
    # m_token = (clip.tokenizer.end_token, 1.0) if  clip.tokenizer.pad_with_end else (0,1.0)
    # using the comma token as a masking token seems to work better than aos tokens for SD 1.x
    m_token = (m_token, 1.0)

    masked_tokens = []

    masked_current = tokens
    for i in range(len(w)):
        if w[i] >= 1:
            continue
        masked_current = mask_inds(masked_current, np.where(w_inv == i)[0], m_token)
        masked_tokens.extend(masked_current)

    embs = batched_clip_encode(masked_tokens, length, encode_func, len(tokens))
    embs = torch.cat([base_emb, embs])
    w = w[w <= 1.0]
    w_mix = np.diff([0] + w.tolist())
    w_mix = torch.tensor(w_mix, dtype=embs.dtype, device=embs.device).reshape((-1, 1, 1))

    weighted_emb = (w_mix * embs).sum(axis=0, keepdim=True)
    return weighted_emb, masked_current, weighted_emb[0, length - 1 : length, :]


def scale_emb_to_mag(base_emb, weighted_emb):
    norm_base = torch.linalg.norm(base_emb)
    norm_weighted = torch.linalg.norm(weighted_emb)
    embeddings_final = (norm_base / norm_weighted) * weighted_emb
    return embeddings_final


def recover_dist(base_emb, weighted_emb):
    fixed_std = (base_emb.std() / weighted_emb.std()) * (weighted_emb - weighted_emb.mean())
    embeddings_final = fixed_std + (base_emb.mean() - fixed_std.mean())
    return embeddings_final


def perp_encode_new(weights, unweighted_embs, empty_embs):
    unweighted, unweighted_pooled = unweighted_embs
    zero, zero_pooled = empty_embs

    weights = weights_like(weights, unweighted)

    if zero.shape != unweighted.shape:
        zero = zero.repeat(1, unweighted.shape[1] // zero.shape[1], 1)

    perp = (
        torch.mul(zero, unweighted).sum(dim=-1, keepdim=True) / (unweighted.norm(dim=-1, keepdim=True) ** 2)
    ) * unweighted

    over1 = weights.abs() > 1.0
    result = unweighted + weights * perp
    result[~over1] = (unweighted - (1 - weights) * perp)[~over1]
    result[weights == 0.0] = zero[weights == 0.0]

    return result, unweighted_pooled


def advanced_encode_from_tokens(
    tokenized,
    token_normalization,
    weight_interpretation,
    encode_func,
    m_token=266,
    length=77,
    w_max=1.0,
    return_pooled=False,
    apply_to_pooled=False,
    **extra_args
):
    tokens = [[t for t, _, _ in x] for x in tokenized]
    weights = [[w for _, w, _ in x] for x in tokenized]
    word_ids = [[wid for _, _, wid in x] for x in tokenized]

    for op in token_normalization.split("+"):
        op = op.strip()
        if op == "length":
            # distribute down/up weights over word lengths
            weights = divide_length(word_ids, weights)
        if op == "mean":
            weights = shift_mean_weight(word_ids, weights)

    pooled = None

    if weight_interpretation == "comfy":
        weighted_tokens = [[(t, w) for t, w in zip(x, y)] for x, y in zip(tokens, weights)]
        weighted_emb, pooled_base = encode_func(weighted_tokens)
        pooled = pooled_base
    else:
        unweighted_tokens = [[(t, 1.0) for t, _, _ in x] for x in tokenized]
        base_emb, pooled_base = encode_func(unweighted_tokens)

    if weight_interpretation == "A1111":
        weighted_emb = base_emb * weights_like(weights, base_emb)  # from_zero
        weighted_emb = (base_emb.mean() / weighted_emb.mean()) * weighted_emb  # renormalize
        pooled = pooled_base

    if weight_interpretation == "compel":
        pos_tokens = [[(t, w) if w >= 1.0 else (t, 1.0) for t, w in zip(x, y)] for x, y in zip(tokens, weights)]
        weighted_emb, _ = encode_func(pos_tokens)
        weighted_emb, _, pooled = down_weight(pos_tokens, weights, word_ids, weighted_emb, length, encode_func)

    if weight_interpretation == "comfy++":
        weighted_emb, tokens_down, _ = down_weight(unweighted_tokens, weights, word_ids, base_emb, length, encode_func)
        weights = [[w if w > 1.0 else 1.0 for w in x] for x in weights]
        # unweighted_tokens = [[(t,1.0) for t, _,_ in x] for x in tokens_down]
        embs, pooled = from_masked(unweighted_tokens, weights, word_ids, base_emb, length, encode_func)
        weighted_emb += embs

    if weight_interpretation == "down_weight":
        weights = scale_to_norm(weights, word_ids, w_max)
        weighted_emb, _, pooled = down_weight(unweighted_tokens, weights, word_ids, base_emb, length, encode_func)

    if weight_interpretation == "perp":
        weighted_emb, pooled = perp_encode_new(
            weights, (base_emb, pooled_base), encode_func(extra_args["tokenizer"].tokenize_with_weights(""))
        )

    if return_pooled:
        if apply_to_pooled:
            return weighted_emb, pooled
        else:
            return weighted_emb, pooled_base
    return weighted_emb, None
