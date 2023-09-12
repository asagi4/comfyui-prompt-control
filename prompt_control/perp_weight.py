import torch


# Copied and adapted from https://github.com/bvhari/ComfyUI_PerpWeight/blob/main/clipperpweight.py
def perp_encode(clip, tokens):
    empty_tokens = clip.tokenize("")
    sdxl_flag = False
    if isinstance(empty_tokens, dict):
        sdxl_flag = True

    if sdxl_flag:
        max_tokens = len(tokens["l"][0])
        empty_cond, empty_cond_pooled = clip.encode_from_tokens(empty_tokens, return_pooled=True)
        unweighted_tokens = {}
        unweighted_tokens["l"] = [[(t, 1.0) for t, _ in x] for x in tokens["l"]]
        unweighted_tokens["g"] = [[(t, 1.0) for t, _ in x] for x in tokens["g"]]
        unweighted_cond, unweighted_pooled = clip.encode_from_tokens(unweighted_tokens, return_pooled=True)

        cond = torch.clone(unweighted_cond)
        empty_cond, _ = equalize(empty_cond, unweighted_cond)
        for i in range(unweighted_cond.shape[0]):
            for j in range(unweighted_cond.shape[1]):
                weight_l = tokens["l"][j // max_tokens][j % max_tokens][1]
                if weight_l != 1.0:
                    token_vector_l = unweighted_cond[i][j][:768]
                    zero_vector_l = empty_cond[0][j][:768]
                    perp_l = (
                        (torch.mul(zero_vector_l, token_vector_l).sum()) / (torch.norm(token_vector_l) ** 2)
                    ) * token_vector_l
                    cond[i][j][:768] = token_vector_l + (weight_l * perp_l)

                weight_g = tokens["g"][i][j][1]
                if weight_g != 1.0:
                    token_vector_g = unweighted_cond[i][j][768:]
                    zero_vector_g = empty_cond[0][j][768:]
                    perp_g = (
                        (torch.mul(zero_vector_g, token_vector_g).sum()) / (torch.norm(token_vector_g) ** 2)
                    ) * token_vector_g
                    cond[i][j][768:] = token_vector_g + (weight_g * perp_g)
    else:
        max_tokens = len(tokens[0])
        empty_cond, empty_cond_pooled = clip.encode_from_tokens(empty_tokens, return_pooled=True)
        unweighted_tokens = [[(t, 1.0) for t, _ in x] for x in tokens]
        unweighted_cond, unweighted_pooled = clip.encode_from_tokens(unweighted_tokens, return_pooled=True)

        cond = torch.clone(unweighted_cond)
        empty_cond, _ = equalize(empty_cond, unweighted_cond)
        for i in range(unweighted_cond.shape[0]):
            for j in range(unweighted_cond.shape[1]):
                weight = tokens[j // max_tokens][j % max_tokens][1]
                if weight != 1.0:
                    token_vector = unweighted_cond[i][j]
                    zero_vector = empty_cond[0][j]
                    perp = (
                        (torch.mul(zero_vector, token_vector).sum()) / (torch.norm(token_vector) ** 2)
                    ) * token_vector
                    cond[i][j] = token_vector + (weight * perp)
    return cond, unweighted_pooled

