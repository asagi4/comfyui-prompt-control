# Attention Couple

NOTE: This is still considered an experimental feature, so the syntax may change.

Attention Couple is an attention-based implementation of regional prompting. it is faster and often more flexible than latent-based masking.

The implementation is based on the one by [pamparamm](https://github.com/pamparamm/ComfyUI-ppm.git), modified to use ComfyUI's hook system. This enables it to work with prompt scheduling.

By default, the implementation produces slightly different results from Pamparamm's implementation because ComfyUI will only run the hook for conds that have it attached and can't batch negative conditionings.

As a consequence of this, however, you can also use `ATTN()` in your negative prompt, and it will work correctly.

To enable batching negative prompts, run your positive and negative prompt through the `PPCAttentionCoupleBatchNegative` node. This will make the outputs identical to pamparamm's implementation and will also improve performance. It will fall back to the default behaviour in cases where batching can't be done, so it should always be safe to use.


## Syntax

See also the main syntax documentation for `MASK` etc.

### ATTN: Trigger Attention Couple

Use `ATTN()` to mark a prompt to be used with Attention Couple. `ATTN()` needs to be combined with either `MASK()` or `IMASK()` to work correctly.

If no mask is specified, an implicit `MASK()` is assumed.

For attention masking to take effect, you need at least two prompt segments with the `ATTN()` marker (separated with `AND`). A single prompt with `ATTN()` will simply ignore the marker.

For the first prompt (and the first prompt only) you can also use `FILL()` to automatically mask all parts not masked by other prompt segments.

For example:
```
dog FILL() ATTN() AND cat MASK(0.5 1) ATTN()
```

If typing `ATTN() MASK()` feels bothersome, try the following macro:
```
DEF(AM=ATTN() MASK($1))
```
and then use it like `MASK`: `AM(0 1, 0.5 1)`
