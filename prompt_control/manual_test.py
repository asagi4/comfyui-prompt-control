import main
import nodes
import prompt_control.adv_encode

(l,) = nodes.CLIPLoader.load_clip(None, "clip_l.safetensors")
(t5,) = nodes.CLIPLoader.load_clip(None, "t5base.safetensors")


def adv(t, text, style="A1111", norm="none", **kwargs):
    c = t.tokenize(text, return_word_ids=True)
    if t is t5:
        te = t.patcher.model.t5base.encode_token_weights
        token = t.tokenizer.clip_t5base
        tok = c["t5base"]
    else:
        te = t.patcher.model.clip_l.encode_token_weights
        token = t.tokenizer.clip_l
        tok = c["l"]
    return prompt_control.adv_encode.advanced_encode_from_tokens(tok, norm, style, te, tokenizer=token)
