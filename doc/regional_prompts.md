# Regional prompting

This section documents the masking functionality of `PCTextEncode`

See also [Attention Couple](/doc/attention_couple.md)

Remember that when using the lazy nodes, prompt scheduling applies to masks as well, so you can change or enable/disable regional prompts at any point during sampling.

## Behaviour

For each prompt separated by `AND`, you can specify either latent masks or an area.

- When masked, ComfyUI generates the model output using the **full latent** as the input, and then applies the mask to the output before adding it to your latent for the next step.
- When an area is specified, ComfyUI generates a separate model output using the **part of the latent specified by the area** and then composites it into the full latent afterwards.
- You can have *both* an AREA and a MASK specified, in which case the mask is applied to the latent specified by the AREA.

For example, consider a 1024 by 1024 (width x height) generation:

- `cat MASK(0 0.5, 0 1) AND dog MASK(0.5 1, 0 1)` generates two outputs at 1024x1024 for "dog" and "cat", then masks half of them off and adds the results together. The following step still see both the dog and the cat from the previous step, so they may blend slightly.

- `cat AREA(0 0.5, 0 1) AND dog AREA(0.5 1, 0 1)` generates two completely separate outputs at **512**x1024 and then composites them together into the 1024x1024 latent. Because the areas do not overlap, the generation for `cat` will not see the output of `dog` and vice versa in subsequent steps as long as the area restriction is in effect.

## MASK, IMASK and AREA

You can use `MASK(x1 x2, y1 y2, weight, op)` to specify a region mask for a prompt. The values are specified as a percentage with a float between `0` and `1`, or as absolute pixel values (these can't be mixed). `1` will be interpreted as a percentage instead of a pixel value.

Multiple `MASK` or `IMASK` calls will be composited together using ComfyUI's `MaskComposite` node, using `op` as the `operation` parameter (defaulting to `multiply`).

Similarly, you can use `AREA(x1 x2, y1 y2, weight)` to specify an area for the prompt (see ComfyUI's area composition examples). The area is calculated by ComfyUI relative to your latent size.

### Custom masks: IMASK and `PCAddMaskToCLIP`

You can attach custom masks to a `CLIP` with the `PC: Attach Mask` nodes and then refer to those masks in the prompt using `IMASK(index, weight, op)`. Indexing starts from zero, so 0 is the first attached mask etc. `PCSCheduleAddMasks` ignores empty inputs, so if you only add a mask to the `mask4` input, it will still have index 0.

Applying the nodes multiple times *appends* masks rather than overriding existing ones, so if you need more than 4, you can just use it more than once.

### Behaviour of multiple masks
If multiple `MASK`s are specified, they are combined together with ComfyUI's `MaskComposite` node, with `op` specifying the operation to use (default `multiply`). In this case, the combined mask weight can be set with `MASKW(weight)` (defaults to 1.0).

Masks assume a size of `(512, 512)`, unless overridden with `PC: Configure PCTextEncode` and pixel values will be relative to that. ComfyUI will scale the mask to match the image resolution. You can change it manually by using `MASK_SIZE(width, height)` anywhere in the prompt,

These are handled per `AND`-ed prompt, so in `prompt1 AND MASK(...) prompt2`, the mask will only affect prompt2.

The default values are `MASK(0 1, 0 1, 1)` and you can omit unnecessary ones, that is, `MASK(0 0.5, 0.3)` is `MASK(0 0.5, 0.3 1, 1)`

Note that because the default values are percentages, `MASK(0 256, 64 512)` is valid, but `MASK(0 200)` will raise an error.

Masking does not affect LoRA scheduling unless you set unet weights to 0 for a LoRA.

## FEATHER: Mask operations

When you use `MASK` or `IMASK`, you can also call `FEATHER(left top right bottom)` to apply feathering using ComfyUI's `FeatherMask` node. The values are in pixels and default to `0`.

If multiple masks are used, `FEATHER` is applied *before compositing* in the order they appear in the prompt, and any leftovers are applied to the combined mask. If you want to skip feathering a mask while compositing, just use `FEATHER()` with no arguments.

For example:
```
MASK(1) MASK(2) MASK(3) FEATHER(1) FEATHER() FEATHER(3) weirdmask FEATHER(4)
```

gives you a mask that is a combination of 1, 2 and 3, where 1 and 3 are feathered before compositing and then `FEATHER(4)` is applied to the composite.

The order of the `FEATHER` and `MASK` calls doesn't matter; you can have `FEATHER` before `MASK` or even interleave them.
