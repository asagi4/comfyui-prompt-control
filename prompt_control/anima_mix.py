"""AVG and CAT for Anima.

ConditioningAverage / ConditioningConcat mix only the conditioning tensor and
keep the first cond's dict. Anima keeps its T5 token ids in that dict, and its
LLM adapter uses them as the attention queries over the Qwen hidden states, so
both collapse to the first prompt. Here every piece keeps its own hidden states
and ids; at sampling time each piece goes through the adapter on its own, CAT
pieces are concatenated along the sequence and AVG entries are averaged.
"""

import logging

import comfy.conds
import torch

log = logging.getLogger("comfyui-prompt-control")

MIX_KEY = "pc_anima_mix"
# The adapter output Anima pads every prompt to.
MIN_LENGTH = 512
# Every piece is tokenized as a whole prompt and ends with </s>; only the last one keeps it.
T5_EOS = 1


def avg_fractions(weights: list[float]) -> list[float]:
    """Final share of each entry under PC's left fold.

    weights[i] is the AVG weight between entry i and entry i+1 (the last one is
    unused). Mirrors encode_prompt_segment, including its quirk that a weight of
    exactly 1.0 drops the following entry instead of averaging with it.
    """
    fractions = [1.0]
    w = weights[0]
    for next_w in weights[1:]:
        if w == 1.0:
            fractions.append(0.0)
        else:
            fractions = [f * w for f in fractions] + [1.0 - w]
        w = next_w
    return fractions


def cat_parts(conds_to_cat) -> list[list]:
    """Snapshot the CAT pieces per cond index, before the concat rewrites the first one in place."""
    return [[(conds[i][0], conds[i][1]) for conds in conds_to_cat] for i in range(len(conds_to_cat[0]))]


def attach(result, avg_entries) -> None:
    """Record every piece on the combined conds when they come from Anima's TE.

    result: the conds the AVG fold produced (modified in place).
    avg_entries: [(cat_parts(...), avg weight), ...], one per AVG entry.
    """
    if len(avg_entries) < 2 and all(len(parts) < 2 for parts in avg_entries[0][0]):
        return
    for parts, _ in avg_entries:
        if len(parts) != len(result) or any("t5xxl_ids" not in d for pieces in parts for _, d in pieces):
            return
    fractions = avg_fractions([w for _, w in avg_entries])
    for i in range(len(result)):
        mix = [
            (f, [(hidden, d["t5xxl_ids"], d.get("t5xxl_weights")) for hidden, d in parts[i]])
            for (parts, _), f in zip(avg_entries, fractions, strict=True)
            if f > 0
        ]
        result[i][1] = {**result[i][1], MIX_KEY: mix}


def _pad(t: torch.Tensor, length: int) -> torch.Tensor:
    return torch.nn.functional.pad(t, (0, 0, 0, length - t.shape[1]))


def install() -> None:
    """Wrap Anima.extra_conds once. Conds without MIX_KEY take the original path untouched."""
    try:
        from comfy.model_base import Anima
    except ImportError:
        return
    if getattr(Anima.extra_conds, "_pc_anima_mix", False):
        return
    original = Anima.extra_conds

    def extra_conds(self, **kwargs):
        mix = kwargs.get(MIX_KEY)
        if not mix or not torch.is_inference_mode_enabled():
            return original(self, **kwargs)
        out = None
        mixed = None
        for fraction, pieces in mix:
            seq = []
            for n, (hidden, ids, weights) in enumerate(pieces):
                out = original(self, **{**kwargs, "cross_attn": hidden, "t5xxl_ids": ids, "t5xxl_weights": weights})
                # The adapter yields one row per T5 token; drop Anima's padding before joining.
                length = ids.shape[-1]
                if n < len(pieces) - 1 and int(ids.reshape(-1)[-1]) == T5_EOS:
                    length -= 1
                seq.append(out["c_crossattn"].cond[:, :length])
            entry = torch.cat(seq, dim=1).float()
            entry = _pad(entry, max(MIN_LENGTH, entry.shape[1])) * fraction
            if mixed is None:
                mixed = entry
                continue
            length = max(mixed.shape[1], entry.shape[1])
            mixed = _pad(mixed, length) + _pad(entry, length)
        assert out is not None and mixed is not None
        out["c_crossattn"] = comfy.conds.CONDRegular(mixed.to(out["c_crossattn"].cond.dtype))
        return out

    extra_conds._pc_anima_mix = True  # ty: ignore[unresolved-attribute]
    Anima.extra_conds = extra_conds
    log.debug("Anima AVG/CAT: extra_conds wrapped")
