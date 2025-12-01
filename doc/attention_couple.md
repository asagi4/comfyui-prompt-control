# Attention Couple

NOTE: This is still considered an experimental feature, so the syntax may change.

Attention Couple is an attention-based implementation of regional prompting. it is faster and often more flexible than latent-based masking.

The implementation is based on the one by [pamparamm](https://github.com/pamparamm/ComfyUI-ppm.git), modified to use ComfyUI's hook system. This enables it to work with prompt scheduling.

By default, the implementation produces slightly different results from Pamparamm's implementation because ComfyUI will only run the hook for conds that have it attached and can't batch negative conditionings.

As a consequence of this, however, you can also use `COUPLE` in your negative prompt, and it will work correctly.

To enable batching negative prompts, run your positive and negative prompt through the `PPCAttentionCoupleBatchNegative` node. This will make the outputs identical to pamparamm's implementation and will also improve performance. It will fall back to the default behaviour in cases where batching can't be done, so it should always be safe to use.


## Syntax

See also the [regional prompting documentation](/doc/regional_prompts.md) for information about `MASK` etc.

### COUPLE: Trigger Attention Couple

You can use `COUPLE` to attach attention-coupled prompts to a base prompt:

`base_prompt COUPLE MASK(0 0.5) coupled prompt 1 with mask COUPLE IMASK(0) coupled prompt 2 with custom mask`

as a shortcut, `COUPLE(maskparams)` is expanded to `COUPLE MASK(maskparams)`, so the above prompt can also be written as:

`base_prompt COUPLE(0 0.5) coupled prompt 1 with mask COUPLE IMASK(0) coupled prompt 2 with custom mask`

Behaviour:
- If no mask is specified, an implicit `MASK()` is assumed.

- For the base prompt, you can also use `FILL()` to automatically mask all parts not masked by coupled prompts

- If the base prompt has weight set to zero (ie. ´:0` at the end), then the first coupled prompt with non-zero weight becomes the base prompt.

For example:
```
dog FILL() COUPLE(0.5 1) cat
```

Note that because the generation still sees and diffuses the full latent, attention coupling is not guaranteed to perfectly limit the effect of your prompt to the masked area.
