import torch
import numpy as np
from math import copysign
import logging
import itertools

log = logging.getLogger("comfyui-prompt-control")


def _norm_mag(w, n):
    d = w - 1
    return 1 + np.sign(d) * np.sqrt(np.abs(d) ** 2 / n)
    # return  np.sign(w) * np.sqrt(np.abs(w)**2 / n)


def _grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def batched_clip_encode(tokens, length, encode_func, num_chunks):
    embs = []
    for e in _grouper(32, tokens):
        enc, pooled = encode_func(e)
        enc = enc.reshape((len(e), length, -1))
        embs.append(enc)

    embs = torch.cat(embs)
    embs = embs.reshape((len(tokens) // num_chunks, length * num_chunks, -1))
    return embs


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


def from_masked(tokens, weights, word_ids, base_emb, length, encode_func, m_token=266):
    pooled_base = base_emb[0, length - 1 : length, :]
    wids, inds = np.unique(np.array(word_ids).reshape(-1), return_index=True)
    weight_dict = dict((id, w) for id, w in zip(wids, np.array(weights).reshape(-1)[inds]) if w != 1.0)

    if len(weight_dict) == 0:
        return torch.zeros_like(base_emb), base_emb[0, length - 1 : length, :]

    weight_tensor = torch.tensor(weights, dtype=base_emb.dtype, device=base_emb.device)
    weight_tensor = weight_tensor.reshape(1, -1, 1).expand(base_emb.shape)

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

        m = torch.tensor(m, dtype=base_emb.dtype, device=base_emb.device)
        m = m.reshape(1, -1, 1).expand(base_emb.shape)
        masks.append(m)

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


def down_weight(tokens, weights, word_ids, base_emb, length, encode_func):
    w, w_inv = np.unique(weights, return_inverse=True)

    if np.sum(w < 1) == 0:
        return base_emb, tokens, base_emb[0, length - 1 : length, :]
    # m_token = (clip.tokenizer.end_token, 1.0) if  clip.tokenizer.pad_with_end else (0,1.0)
    # using the comma token as a masking token seems to work better than aos tokens for SD 1.x
    m_token = (266, 1.0)

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


def perp_weight(weights, unweighted_embs, empty_embs):
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

    # Not sure if this is an implementation bug or if this just doesn't make sense with T5
    nans = result.isnan()
    if nans.any():
        log.warning("perp weight returned NaNs (known to happen with T5), replacing with 0")
        result[nans] = 0.0

    return result, unweighted_pooled


def style_comfy(encoder, tokens, **kwargs):
    tokens = encoder.without_word_ids(tokens)
    return encoder.encode_fn(tokens)


def style_a1111(encoder, tokens, **kwargs):
    base_emb, pooled = encoder.base_emb(tokens)
    weighted_emb = base_emb * weights_like(encoder.weights(tokens), base_emb)
    weighted_emb = (base_emb.mean() / weighted_emb.mean()) * weighted_emb  # renormalize
    return weighted_emb, pooled


def style_compel(encoder, tokens, **kwargs):
    pos_tokens = encoder.weighted_with(tokens, lambda w: w if w > 1.0 else 1.0)
    weighted_emb, pooled = encoder.encode_fn(pos_tokens)
    weighted_emb, _, pooled = encoder.down_weight(
        pos_tokens, encoder.weights(tokens), encoder.word_ids(tokens), weighted_emb, pooled
    )
    return weighted_emb, pooled


def style_comfypp(encoder, tokens, **kwargs):
    unweighted_tokens = encoder.unweighted(tokens)
    base_emb, pooled_base = encoder.base_emb(tokens)
    weighted_emb, tokens_down, _ = encoder.down_weight(
        unweighted_tokens, encoder.weights(tokens), encoder.word_ids(tokens), base_emb, pooled_base
    )
    weights = encoder.weights(encoder.weighted_with(tokens, lambda w: w if w > 1.0 else 1.0))
    embs, pooled = encoder.from_masked(
        unweighted_tokens,
        weights,
        encoder.word_ids(tokens),
        base_emb,
        pooled_base,
    )
    weighted_emb += embs

    return weighted_emb, pooled


def style_downweight(encoder, tokens, **kwargs):
    weights = scale_to_norm(encoder.weights(tokens), encoder.word_ids(tokens), encoder.w_max)
    base_emb, pooled_base = encoder.base_emb(tokens)
    weighted_emb, _, pooled = encoder.down_weight(
        encoder.unweighted(tokens), weights, encoder.word_ids(tokens), base_emb, pooled_base
    )

    return weighted_emb, pooled


def style_perp(encoder, tokens, **kwargs):
    zero_emb, zero_pooled = encoder.encode_fn(encoder.tokenizer.tokenize_with_weights(""))
    base_emb, pooled = encoder.base_emb(tokens)
    return perp_weight(encoder.weights(tokens), (base_emb, pooled), (zero_emb, zero_pooled))


def apply_negpip(encoder, emb, pooled, **kwargs):
    original_tokens = kwargs["original_tokens"]
    emb_negpip = torch.empty_like(emb).repeat(1, 2, 1)
    emb_negpip[:, 0::2, :] = emb
    emb_negpip[:, 1::2, :] = emb * weights_like(encoder.signs(original_tokens), emb)
    return emb_negpip, pooled


def norm_length(encoder, tokens, **kwargs):
    word_ids = encoder.word_ids(tokens)
    sums = dict(zip(*np.unique(word_ids, return_counts=True)))
    sums[0] = 1
    tokens = [[(t, _norm_mag(w, sums[id]) if id != 0 else 1.0, id) for (t, w, id) in x] for x in tokens]
    return tokens


def norm_mean(encoder, tokens, **kwargs):
    weights = encoder.weights(tokens)
    word_ids = encoder.word_ids(tokens)
    delta = 1 - np.mean([w for x, y in zip(weights, word_ids) for w, id in zip(x, y) if id != 0])
    tokens = [[(t, w if id == 0 else w + delta, id) for (t, w, id) in x] for x in tokens]
    return tokens


def norm_none(encoder, tokens, **kwargs):
    return tokens


class AdvancedEncoder:
    STYLES = {
        "A1111": style_a1111,
        "comfy": style_comfy,
        "comfy++": style_comfypp,
        "compel": style_compel,
        "down_weight": style_downweight,
        "perp": style_perp,
    }
    NORMALIZATION_OPS = {
        "none": norm_none,
        "length": norm_length,
        "mean": norm_mean,
    }

    @classmethod
    def add_encoder(cls, name, fn):
        cls.STYLES[name] = fn

    def add_normalization_op(cls, name, fn):
        cls.NORMALIZATION_OPS[name] = fn

    @classmethod
    def weighted_with(cls, tokens, fn=id, word_ids=True):
        w = ([(t, fn(w), id) for t, w, id in x] for x in tokens)
        if not word_ids:
            w = cls.without_word_ids(w)
        return list(w)

    @classmethod
    def unweighted(cls, tokens, word_ids=False):
        return cls.weighted_with(tokens, fn=lambda w: 1.0, word_ids=word_ids)

    @classmethod
    def tokens_only(cls, tokens):
        return list([t[0] for t in x] for x in tokens)

    @classmethod
    def weights(cls, tokens):
        return list([t[1] for t in x] for x in tokens)

    @classmethod
    def word_ids(cls, tokens):
        return list([t[2] for t in x] for x in tokens)

    @classmethod
    def signs(cls, tokens):
        return list([copysign(1, t[1]) for t in x] for x in tokens)

    @classmethod
    def without_word_ids(cls, tokens):
        return list([(t, w) for t, w, _ in x] for x in tokens)

    def __init__(self, encode_fn, style, normalization, tokenizer, m_token="+", w_max=1.0, **extra_args):
        self.encode_fn = encode_fn
        self.preprocessors = []
        self.postprocessors = []
        self.tokenizer = tokenizer
        self.extra_args = extra_args
        self.m_token = tokenizer.tokenize_with_weights(m_token)[0][tokenizer.tokens_start]
        self.max_length = tokenizer.max_length if tokenizer.pad_to_max_length else None
        self.w_max = w_max

        if style == "comfy++" and not self.max_length:
            log.warning("comfy++ does not work with tokenizer %s, using default weighting", tokenizer)
            style = "comfy"

        norms = normalization.split("+")
        assert style in self.STYLES, f"Invalid weight interpretation: {style}"
        self.weight_fn = self.STYLES[style]
        for n in norms:
            n = n.strip()
            assert n in self.NORMALIZATION_OPS, f"Invalid normalization: {normalization}"
            self.preprocessors.append(self.NORMALIZATION_OPS[n])

        negpip = extra_args.get("has_negpip")
        if negpip:

            def _encode(t):
                emb, pooled = encode_fn(t)
                return emb[:, 0::2, :], pooled

            self.encode_fn = _encode
            self.preprocessors.insert(lambda encoder, tokens, **kwargs: encoder.weighted_with(tokens, abs))
            self.postprocessors.insert(0, apply_negpip)

    def base_emb(self, tokens):
        unweighted = self.unweighted(tokens)
        return self.encode_fn(unweighted)

    def down_weight(self, tokens, weights, word_ids, base_emb, pooled_base):
        w, w_inv = np.unique(weights, return_inverse=True)

        if np.sum(w < 1) == 0:
            return (
                base_emb,
                tokens,
                (
                    base_emb[0, self.max_length - 1 : self.max_length, :]
                    if (pooled_base is not None and self.max_length)
                    else None
                ),
            )

        masked_current = tokens
        emblist = [base_emb]
        for i in range(len(w)):
            if w[i] >= 1:
                continue
            masked_current = mask_inds(masked_current, np.where(w_inv == i)[0], self.m_token)
            masked, _ = self.encode_fn(masked_current)
            emblist.append(masked)

        embs = torch.cat(emblist)
        w = w[w <= 1.0]
        w_mix = np.diff([0] + w.tolist())
        w_mix = torch.tensor(w_mix, dtype=embs.dtype, device=embs.device).reshape((-1, 1, 1))

        weighted_emb = (w_mix * embs).sum(axis=0, keepdim=True)
        pooled = pooled_base
        if pooled is not None and self.max_length:
            pooled = weighted_emb[0, self.max_length - 1 : self.max_length, :]
        return weighted_emb, masked_current, pooled

    def from_masked(self, tokens, weights, word_ids, base_emb, pooled_base):
        wids, inds = np.unique(np.array(word_ids).reshape(-1), return_index=True)
        weight_dict = dict((id, w) for id, w in zip(wids, np.array(weights).reshape(-1)[inds]) if w != 1.0)

        if len(weight_dict) == 0:
            return torch.zeros_like(base_emb), torch.zeros_like(pooled_base) if pooled_base is not None else None

        weight_tensor = weights_like(weights, base_emb)

        ws = []
        masked_tokens = []
        masks = []

        # create prompts
        for id, w in weight_dict.items():
            masked, m = mask_word_id(tokens, word_ids, id, self.m_token)
            masks.append(weights_like(m, base_emb))
            masked_tokens.extend(masked)

            ws.append(w)

        # TODO: figure out how to get rid of this
        embs = batched_clip_encode(masked_tokens, self.max_length, self.encode_fn, len(tokens))
        masks = torch.cat(masks)

        embs = base_emb.expand(embs.shape) - embs
        if pooled_base is not None and self.max_length:
            pooled = embs[0, self.max_length - 1 : self.max_length, :]
            pooled_start = pooled_base.expand(len(ws), -1)
            ws = torch.tensor(ws).reshape(-1, 1).expand(pooled_start.shape)
            pooled = (pooled - pooled_start) * (ws - 1)
            pooled = pooled.mean(axis=0, keepdim=True)
            pooled = pooled_base + pooled

        if embs.shape[0] != masks.shape[0]:
            embs = embs.repeat(masks.shape[0], 1, 1)
        embs *= masks
        embs = embs.sum(axis=0, keepdim=True)

        return ((weight_tensor - 1) * embs), pooled

    def __call__(self, tokens, apply_to_pooled=False, return_pooled=False):
        normalized_tokens = tokens
        for op in self.preprocessors:
            normalized_tokens = op(self, normalized_tokens)

        emb, pooled = self.weight_fn(self, normalized_tokens, original_tokens=tokens)

        for fn in self.postprocessors:
            emb, pooled = fn(self, emb, pooled, tokens=tokens, original_tokens=tokens)

        if return_pooled:
            if not apply_to_pooled:
                _, pooled = self.base_emb(tokens)
            return emb, pooled
        return emb, None


# For verification
def A1111_renorm(base_emb, weighted_emb):
    embeddings_final = (base_emb.mean() / weighted_emb.mean()) * weighted_emb
    return embeddings_final


def from_zero(weights, base_emb):
    weight_tensor = torch.tensor(weights, dtype=base_emb.dtype, device=base_emb.device)
    weight_tensor = weight_tensor.reshape(1, -1, 1).expand(base_emb.shape)
    return base_emb * weight_tensor


def old_advanced_encode_from_tokens(
    tokenized,
    token_normalization,
    weight_interpretation,
    encode_func,
    m_token=266,
    w_max=1.0,
    return_pooled=False,
    apply_to_pooled=False,
    **extra_args,
):
    length = 77
    tokens = [[t for t, _, _ in x] for x in tokenized]
    weights = [[w for _, w, _ in x] for x in tokenized]
    word_ids = [[wid for _, _, wid in x] for x in tokenized]

    # weight normalization
    # ====================

    # distribute down/up weights over word lengths
    if token_normalization.startswith("length"):
        weights = divide_length(word_ids, weights)

    # make mean of word tokens 1
    if token_normalization.endswith("mean"):
        weights = shift_mean_weight(word_ids, weights)

    # weight interpretation
    # =====================
    pooled = None

    if weight_interpretation in ["comfy", "perp"]:
        weighted_tokens = [[(t, w) for t, w in zip(x, y)] for x, y in zip(tokens, weights)]
        weighted_emb, pooled_base = encode_func(weighted_tokens)
        pooled = pooled_base
    else:
        unweighted_tokens = [[(t, 1.0) for t, _, _ in x] for x in tokenized]
        base_emb, pooled_base = encode_func(unweighted_tokens)

    if weight_interpretation == "A1111":
        weighted_emb = from_zero(weights, base_emb)
        weighted_emb = A1111_renorm(base_emb, weighted_emb)
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

    if return_pooled:
        if apply_to_pooled:
            return weighted_emb, pooled
        else:
            return weighted_emb, pooled_base
    return weighted_emb, None


def advanced_encode_from_tokens(
    tokenized,
    token_normalization,
    weight_interpretation,
    encode_func,
    m_token="+",
    w_max=1.0,
    return_pooled=False,
    apply_to_pooled=False,
    tokenizer=None,
    **extra_args,
):
    if "new+" in weight_interpretation:
        weight_interpretation = weight_interpretation.replace("new+", "")
        enc = AdvancedEncoder(
            encode_func, weight_interpretation, token_normalization, tokenizer, m_token, w_max, **extra_args
        )
        log.info("Using new implementation for %s", weight_interpretation)
        return enc(tokenized, return_pooled=return_pooled, apply_to_pooled=apply_to_pooled)
    else:
        return old_advanced_encode_from_tokens(tokenized, token_normalization, weight_interpretation, encode_func, 266, return_pooled=return_pooled, apply_to_pooled=apply_to_pooled)
