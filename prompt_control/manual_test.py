import main
import nodes
import prompt_control.adv_encode

(l,) = nodes.CLIPLoader.load_clip(None, "clip_l.safetensors")
(t5,) = nodes.CLIPLoader.load_clip(None, "t5base.safetensors")

id(main)  # get rid of warning


def adv(t, text, style="A1111", norm="none", new=True, **kwargs):
    c = t.tokenize(text, return_word_ids=True)
    if new:
        style = "new+" + style
    if t is t5:
        te = t.patcher.model.t5base.encode_token_weights
        token = t.tokenizer.clip_t5base
        tok = c["t5base"]
    else:
        te = t.patcher.model.clip_l.encode_token_weights
        token = t.tokenizer.clip_l
        tok = c["l"]
    return prompt_control.adv_encode.advanced_encode_from_tokens(tok, norm, style, te, tokenizer=token)


def adv_all(t, text, styles=[], **kwargs):
    r = []
    for s in styles or prompt_control.adv_encode.AdvancedEncoder.STYLES:
        print("Testing", s, kwargs)
        r.append([s, adv(t, text, style=s, **kwargs)])
    return r


def replacenan(t):
    t[t.isnan()] = 42.123321
    return t


def adv_equal(t, text, **kwargs):
    old = adv_all(t, text, new=False, **kwargs)
    new = adv_all(t, text, new=True, **kwargs)
    r = {}
    for i, o in enumerate(old):
        n = new[i]
        r[n[0]] = (replacenan(n[1][0]) == replacenan(o[1][0])).all()
    return r
