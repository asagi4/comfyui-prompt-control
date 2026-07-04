# Attention Couple

Attention Couple is an attention-based implementation of regional prompting. it is faster and often more flexible than latent-based masking.

The implementation is based on the one by [pamparamm](https://github.com/pamparamm/ComfyUI-ppm.git), modified to use ComfyUI's hook system. This enables it to work with prompt scheduling.

By default, the implementation produces slightly different results from Pamparamm's implementation because ComfyUI will only run the hook for conds that have it attached and can't batch negative conditionings.

As a consequence of this, however, you can also use `COUPLE` in your negative prompt, and it will work correctly.

To enable batching negative prompts, run your positive and negative prompt through the `PPCAttentionCoupleBatchNegative` node. This will make the outputs identical to pamparamm's implementation and will also improve performance. It will fall back to the default behaviour in cases where batching can't be done, so it should always be safe to use.

## Anima

There is a **very experimental** port of pamparamm's Anima support for Attention Couple in Prompt Control. Because ComfyUI lacks the built-in schedulable hooks required, you must first patch your model with `PC: Anima Attention Couple Model Patch` in addition to using `COUPLE` as usual.

The code was hacked together with minimal thought, so expect bugs and misbehaviour. The port is also currently *not* compatible with NegPIP.

## Syntax

See also the [regional prompting documentation](/doc/regional_prompts.md) for information about `MASK` etc.

### COUPLE: Trigger Attention Couple

You can use `COUPLE` to attach attention-coupled prompts to a base prompt:

For example:
```
dog FILL() COUPLE(0.5 1) cat
```

The full syntax looks as follows (to use `IMASK` you need to attach a custom mask)

`base_prompt COUPLE MASK(0 0.5) coupled prompt 1 with mask COUPLE IMASK(0) coupled prompt 2 with custom mask`

as a shortcut, `COUPLE(maskparams)` is expanded to `COUPLE MASK(maskparams)`, so the above prompt can also be written as:

`base_prompt COUPLE(0 0.5) coupled prompt 1 with mask COUPLE IMASK(0) coupled prompt 2 with custom mask`

Behaviour:
- If no mask is specified, an implicit `MASK()` is assumed, meaning that the prompt affects the entire image.

- For the base prompt, you can use `FILL()` to automatically mask all parts not masked by other coupled prompts

- If the base prompt has weight set to zero (ie. ´:0` at the end), then the first coupled prompt with non-zero weight becomes the base prompt:

```
disabled prompt :0 COUPLE new base prompt COUPLE coupled prompt
```

You can also schedule the weight normally: `prompt :[1:0:0.35]`

> ![NOTE]
> Note that because the generation still sees and diffuses the full latent, attention coupling is not guaranteed to perfectly limit the effect of your prompt to the masked area.
