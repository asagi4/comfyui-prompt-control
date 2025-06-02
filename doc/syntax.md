# Prompt Control Syntax

If you're viewing this on GitHub, I recommend opening the outline by clicking the button in the top right corner of the text view (it is annoyingly easy to miss).

Scheduling syntax is similar to A1111, but only fractions are supported for steps. LoRAs are scheduled by including them in a scheduling expression.

```
a [large::0.1] [cat|dog:0.05] [<lora:somelora:0.5:0.6>::0.5]
[in a park:in space:0.4]
```

## Scheduled prompts

There are two forms of scheduled prompts.

### Basic scheduling expressions
Basic expressions take the form `[before:after:X]` where `X` is the switch point, a decimal number between 0.0 and 1.0 inclusive, representing 0 to 100% of timesteps. Either prompt can also be empty.
For example:
```
a [red:blue:0.5] cat
```
switches from `a red cat` to `a blue cat` at 0.5. `before` and `after` can be arbitrary prompts (`after` can also be empty), including other scheduling expressions, allowing nesting:
```
a [red:[blue::0.7]:0.5] cat
```

switches from `a red cat` to `a blue cat` at 0.5 and to `a cat` at 0.7

For convenience `[cat:0.5]` is equivalent to `[:cat:0.5]` meaning it switches from empty to `cat` at 0.5.

### Range expressions

The most general form of a schedule is a range expression: For example, in `[before:during:after:0.3,0.7]`, The prompt be `a before` until 0.3, `a during` until 0.7, and then `a after`. This form is equivalent to `[before:[during:after:0.7]:0.3]`

For convenience, `[during:0.1,0.4]` is equivalent to `[:during::0.1,0.4]` and `[during:after:0.1,0.4]` is equivalent to `[:during:after:0.1,0.4]`.

`[before:during:after:0.1]` is the same as `[before:during:after:0.1,1.0]` which is same as `[before:during:0.1]`


### Using step numbers with the Advanced nodes

If you provide a non-zero value to `num_steps` to the `Advanced` versions of the scheduling nodes, you will be able to use step numbers in prompts.

For now, a value between 0 and 1.0 will be interpreted as a percentage if it contains a ., and as an absolute step otherwise.

This is just syntactic sugar. Behind the scenes, the values are converted to percentages and have normal ComfyUI scheduling behaviour.

## Tag selection
Using the `FilterSchedule` node, in addition to step percentages, you can use a *tag* to select part of an input:
```
a large [dog:cat<lora:catlora:0.5>:SECOND_PASS]
```
Set the `tags` parameter in the `FilterSchedule` node to filter the prompt. If the tag matches any tag `tags` (comma-separated), the second option is returned (`cat`, in this case, with the LoRA). Otherwise, the first option is chosen (`dog`, without LoRA).

the values in `tags` are case-insensitive, but the tags in the input **must** be uppercase A-Z and underscores only, or they won't be recognized. That is, `[dog:cat:hr]` will not work.

For example, a prompt
```
a [black:blue:X] [cat:dog:Y] [walking:running:Z] in space
```
with `tags` `x,z` would result in the prompt `a blue cat running in space`

The three prompt form `[a:b:c:TAG]` is parsed, but ignores `b` and is equivalent to `[a:c:TAG]`.

## LoRA Scheduling
When using the lazy graph building nodes, LoRAs can be scheduled by referring to them in a scheduling expression, like so:

`<lora:fulllora:1> [<lora:partialora:1>::0.5]`

This will schedule `fulllora` for the entire duration of the prompt and `partiallora` until half of sampling is complete.

You can refer to LoRAs by using the filename without extension and subdirectories will also be searched. For example, `<lora:cats:1>`. will match both `cats.safetensors` and `sd15/animals/cats.safetensors`. If there are multiple LoRAs with the same name, the first match will be loaded.

Alternatively, the name can include the full directory path relative to ComfyUI's search paths, without extension: `<lora:XL/sdxllora:0.5>`. In this case, the *full* path must match.

If no match is found, the node will try to replace spaces with underscores and search again. That is, `<lora:cats and dogs:1>` will find `cats_and_dogs.safetensors`. This helps with some autocompletion scripts that replace underscores with spaces.

Finally, you can give the exact path (including the extension) as shown in `LoRALoader`.


## Alternating

Alternating syntax is `[a|b:pct_steps]`, causing the prompt to alternate every `pct_steps`. `pct_steps` defaults to 0.1 if not specified. You can also have more than two options.


## Sequences

The syntax `[SEQ:a:N1:b:N2:c:N3]` is shorthand for `[a:[b:[c::N3]:N2]:N1]` ie. it switches from `a` to `b` to `c` to nothing at the specified points in sequence.

Might be useful with Jinja templating (see https://github.com/asagi4/comfyui-utility-nodes). For example:
```
[SEQ<% for x in steps(0.1, 0.9, 0.1) %>:<lora:test:<= sin(x*pi) + 0.1 =>>:<= x =><% endfor %>]
```
generates a LoRA schedule based on a sinewave

# Basic prompt syntax

This syntax is also available in outside scheduled with the `PCTextEncode` node, where applicable.

## Combining prompts

### AND

`AND` can be used to create "prompt segments". By default, it works as if you had combined the different prompts with `ConditioningCombine`. 

It is also used with regional prompting to separate different prompts; see `MASK` and `ATTN` below.

Prompts can have a weight at the end:
```
cat :1 AND dog :2
```
`AND` is processed after schedule parsing, so you can change the weight mid-prompt: `cat:[1:2:0.5] AND dog`

The weight defaults to 1. If a prompt's weight is set to 0, it's **skipped entirely.** This can be useful when scheduling to completely disable a prompt:

```
cat [\:0::0.5] AND dog
```
Note that the `:` needs to be escaped with a `\` or it will be interpreted as scheduling syntax.

## Note about processing order

Prompt operators are processed in the following order, meaning that all features "below" another can be affected by the feature above it. That is, `BREAK` can go inside a `TE()` call, but not `AND` or `CAT`.

- DEF macros are expanded
- Scheduling is expanded
- Prompts are split by AND
- Most functions (like STYLE, MASK) and cutoffs are evaluated
- prompts are split by AVG()
- prompts are split by CAT
- the TE() function is evaluated to set per-encoder prompts
- BREAK is evaluated
- Everything else

## Functions

There are some "functions" that can be included in a prompt to affect how it is interpreted.

Functions have the form `FUNCNAME(param1, param2, ...)`. How parameters are interpreted is up to the function. 

In general, function parameters will have default values that are used if the parameter is left empty.

Note: Whitespace is usually *not* stripped from string parameters by default. Commas can be escaped with `\,`

Like `AND`, functions are parsed after regular scheduling syntax has been expanded, allowing things like `[AREA:MASK:0.3](...)`, in case that's somehow useful.

### BREAK
The keyword `BREAK` causes the prompt to be tokenized in separate chunks, padding each chunk to the text encoder's maximum size before encoding.

For some text encoders (like t5), this operation doesn't really make sense and BREAKs are simply ignored.

### CAT

`CAT` encodes each prompt separately before concatenating the resulting tensors into a single conditioning. It behaves identically to ComfyUI's `ConditioningConcat`.

### AVG()

`prompt1 AVG(weight) prompt2` encodes prompt1 and prompt2 separately, and then combines them using `ConditioningAverage`. The default for `weight` is `0.5`.

`AVG` is processed before `BREAK` but after `AND`

`p1 AVG() p2 AVG() p3` combines `p1` and `p2` first, then combines the result with `p3`.

## Prompt weighting (also known as "Advanced CLIP Encode")

### STYLE

Use the syntax `STYLE(weight_interpretation, normalization)` in a prompt to affect how prompts are interpreted.

The weight interpretations available are:
  - comfy (default)
  - comfy++
  - compel
  - down_weight
  - A1111
  - perp

Normalizations are:
  - none (default)
  - length
  - mean

The normalization calculations are independent operations and you can combine them with `+`, eg `STYLE(A1111, length+mean)` or `STYLE(comfy, mean+length)`, or even something silly like `STYLE(perp, mean+length+mean+length)`

The style can be specified separately for each AND:ed prompt, but the first prompt is special; later prompts will "inherit" it as default. For example:

```
STYLE(A1111) a (red:1.1) cat with (brown:0.9) spots and a long tail AND an (old:0.5) dog AND a (green:1.4) (balloon:1.1)
```
will interpret everything as A1111, but
```
a (red:1.1) cat with (brown:0.9) spots and a long tail AND STYLE(A1111) an (old:0.5) dog AND a (green:1.4) (balloon:1.1)
```
Will interpret the first one using the default ComfyUI behaviour, the second prompt with A1111 and the last prompt with the default again

### SDXL: Configure SDXL prompting parameters

The nodes do not treat SDXL models specially, but there are some utilities that enable SDXL specific functionality.

You can use the function `SDXL(width height, target_width target_height, crop_w crop_h)` to set SDXL prompt parameters. `SDXL()` is equivalent to `SDXL(1024 1024, 1024 1024, 0 0)` unless the default values have been overridden by `PCScheduleSettings`.

### TE: Per-encoder prompts for multi-encoder models

You can specify per-encoder prompts using the `TE` function. The syntax is as follows:
`TE(encoder_name=prompt)`. Whitespace surrounding the prompt and encoder name are ignored.

For example:
```
TE(l=cat) TE(g = (dog:1.1)) TE(t5xxl=tiger)
```
The keys to use depend on what key ComfyUI uses for the encoder; for example `l` for CLIP L, `g` for CLIP G, and `t5xxl` for T5 XXL (Flux text encoder).

Use `TE(help)` to print a help text listing available keys.

Things to note:
- If you set a prompt with `TE`, it will override the prompt outside the function for the specified text encoder.
- Multiple instances of `TE` are joined with a space. That is, `TE(l=foo)TE(l=bar)` is the same as `TE(l=foo bar)`
- `AND` and `BREAK` are processed before `TE`, so they do not do anything sensible; `TE(l=foo AND bar)` will parse as two prompts `TE(foo` and `bar)`. `SHIFT`, `SHUFFLE` and `OLDBREAK` do work, however.

### SHUFFLE and SHIFT: Create prompt permutations

Default parameters: `SHUFFLE(seed=0, separator=,, joiner=,)`, `SHIFT(steps=0, separator=,, joiner=,)`

`SHIFT` moves elements to the left by `steps`. The default is 0 so `SHIFT()` does nothing
`SHUFFLE` generates a random permutation with `seed` as its seed.

These functions are applied to each prompt chunk **after** `BREAK`, `AND` etc. have been parsed. The prompt is split by `separator`, the operation is applied, and it's then joined back by `joiner`.

Multiple instances of these functions are applied in the order they appear in the prompt.

**NOTE** To avoid breaking emphasis syntax, the functions ignore any separators inside parentheses

For example:
- `SHIFT(1) cat, dog, tiger, mouse` does a shift and results in `dog, tiger, mouse, cat`. (whitespace may vary)
- `SHIFT(1,;) cat, dog ; tiger, mouse` results in `tiger, mouse, cat, dog`
- `SHUFFLE() cat, dog, tiger, mouse` results in `cat, dog, mouse, tiger`
- `SHUFFLE() SHIFT(1) cat, dog, tiger, mouse` results in `dog, mouse, tiger, cat`

- `SHIFT(1) cat,dog BREAK tiger,mouse` results in `dog,cat BREAK tiger,mouse`
- `SHIFT(1) cat, dog AND SHIFT(1) tiger, mouse` results in `dog, cat BREAK mouse, tiger`

Whitespace is *not* stripped and may also be used as a joiner or separator
- `SHIFT(1,, ) cat,dog` results in `dog cat`

### NOISE: Add noise to a prompt

The function `NOISE(weight, seed)` adds some random noise into the cond tensor. The seed is optional, and if not specified, the global RNG is used. `weight` should be between 0 and 1.

The usefulness of this is questionable, but it wasn't difficult to implement, so here it is.


## Regional prompting

See also [Attention Couple](#attention-couple) below

### MASK, IMASK and AREA

You can use `MASK(x1 x2, y1 y2, weight, op)` to specify a region mask for a prompt. The values are specified as a percentage with a float between `0` and `1`, or as absolute pixel values (these can't be mixed). `1` will be interpreted as a percentage instead of a pixel value.

Multiple `MASK` or `IMASK` calls will be composited together using ComfyUI's `MaskComposite` node, using `op` as the `operation` parameter (defaulting to `multiply`).

Similarly, you can use `AREA(x1 x2, y1 y2, weight)` to specify an area for the prompt (see ComfyUI's area composition examples). The area is calculated by ComfyUI relative to your latent size.

### Custom masks: IMASK and `PCAddMaskToCLIP`

You can attach custom masks to a `CLIP` with the `PC: Attach Mask` nodes and then refer to those masks in the prompt using `IMASK(index, weight, op)`. Indexing starts from zero, so 0 is the first attached mask etc. `PCSCheduleAddMasks` ignores empty inputs, so if you only add a mask to the `mask4` input, it will still have index 0.

Applying the nodes multiple times *appends* masks rather than overriding existing ones, so if you need more than 4, you can just use it more than once.

### Behaviour of masks
If multiple `MASK`s are specified, they are combined together with ComfyUI's `MaskComposite` node, with `op` specifying the operation to use (default `multiply`). In this case, the combined mask weight can be set with `MASKW(weight)` (defaults to 1.0).

Masks assume a size of `(512, 512)`, unless overridden with `PC: Configure PCTextEncode` and pixel values will be relative to that. ComfyUI will scale the mask to match the image resolution. You can change it manually by using `MASK_SIZE(width, height)` anywhere in the prompt,

These are handled per `AND`-ed prompt, so in `prompt1 AND MASK(...) prompt2`, the mask will only affect prompt2.

The default values are `MASK(0 1, 0 1, 1)` and you can omit unnecessary ones, that is, `MASK(0 0.5, 0.3)` is `MASK(0 0.5, 0.3 1, 1)`

Note that because the default values are percentages, `MASK(0 256, 64 512)` is valid, but `MASK(0 200)` will raise an error.

Masking does not affect LoRA scheduling unless you set unet weights to 0 for a LoRA.

### FEATHER: Mask operations

When you use `MASK` or `IMASK`, you can also call `FEATHER(left top right bottom)` to apply feathering using ComfyUI's `FeatherMask` node. The values are in pixels and default to `0`.

If multiple masks are used, `FEATHER` is applied *before compositing* in the order they appear in the prompt, and any leftovers are applied to the combined mask. If you want to skip feathering a mask while compositing, just use `FEATHER()` with no arguments.

For example:
```
MASK(1) MASK(2) MASK(3) FEATHER(1) FEATHER() FEATHER(3) weirdmask FEATHER(4)
```

gives you a mask that is a combination of 1, 2 and 3, where 1 and 3 are feathered before compositing and then `FEATHER(4)` is applied to the composite.

The order of the `FEATHER` and `MASK` calls doesn't matter; you can have `FEATHER` before `MASK` or even interleave them.

## Cutoff

NOTE: Cutoff syntax might change at some point; it's pretty clunky.

`PCTextEncode` reimplements cutoff from [ComfyUI Cutoff](https://github.com/BlenderNeko/ComfyUI_Cutoff).

The syntax is
```
a group of animals, [CUT:white cat:white], [CUT:brown dog:brown:0.5:1.0:1.0:_]
```
You should read the prompt as `a group of animals, white cat, brown dog`, but CUT causes the tokens in `target_tokens` to be masked off from the base prompt in `region_text`, so that their effect can be isolated, and you're less likely to get brown cats or white dogs.

Target tokens are treated individually, separated by space, for example, `[CUT:green apple, red apple, green leaf:green apple]` will mask *both* greens and the apple, giving you `+ +, red +, + leaf`. To mask out just `green apple`, use `[CUT:green apple, red apple:green_apple]` which will result in a masked prompt of `+ +, red apple`. Escape `_` with a `\`.

the parameters in the `CUT` section are `region_text:target_tokens:weight;strict_mask:start_from_masked:padding_token` of which only the first two are required. The default values are `weight=1.0`, `strict_mask=1.0` `start_from_masked=1.0`, `padding_token=+`

If `strict_mask`, `start_from_masked` or `padding_token` are specified in more than one CUT, the *last* one becomes the default for any CUTs afterwards that do not explicitly set the parameters. For example, in:

`[CUT:white cat:white:0.5] and [CUT:black parrot, flying:black:1.0:0.5] and [CUT:green apple:green]`

`white cat` will a weight of 0.5, and 1.0 for all parameters, and `black parrot` and `green apple` will *both* have a `strict_mask` parameter of 0.5.

The parameters affect how the masked and unmasked prompts are combined to produce the final embedding. Just play around with them.

## Miscellaneous
- `<emb:xyz>` is alternative syntax for `embedding:xyz` to work around a syntax conflict with `[embedding:xyz:0.5]` which is parsed as a schedule that switches from `embedding` to `xyz`.

# Experimental features

Experimental features are unstable and may disappear or change without warning.

## DEF: Lightweight prompt macros

You can define "prompt macros" by using `DEF`.  Macros are expanded before any other parsing takes place. The expansion continues until no further changes occur. Recursion will raise an error.

`PCLazyTextEncode` and `PCLazyLoraLoader` expand macros, but `PCTextEncode` **does not**. If you need to expand macros for a single prompt, use `PCMacroExpand`

```
DEF(MYMACRO=this is a prompt)
[(MYMACRO:0.6):(MYMACRO:1.1):0.5]
```
is equivalent to
```
[(this is a prompt:0.5):(this is a prompt:1.1):0.5]
```
### Macro parameters
It's also possible to give parameters to a macro:
```
DEF(MYMACRO=[(prompt $1:$2):(prompt $1:$3):$4])
MYMACRO(test; 1.1; 0.7; 0.2)
```
gives
```
[(prompt test:1.1):(prompt test:0.7):0.2]
```
in this form, the variables $N (where N is any number corresponding to a positional parameter) will be replaced with the given parameter. The parameters must be separated with a semicolon, and can be empty.

You can also optionally specify default values:

```
DEF(MACRO(example; 0; 1)=[$1:$2,$3])
MACRO MACRO(test; 0.2)
```
gives
```
[example:0,1] [test:0.2,1]
```

```
DEF(MACRO() = [a:$1:0.5])
```
sets the default value of `$1` to an empty string.

### Unspecified parameters in macros

Unspecified parameters (either via defaults or explicitly given) will not be substituted. Compare:

```
DEF(mything=a "$1" b "$2")
mything
mything()
mything(A)
```

gives

```
a "$1" b "$2"
a "" b "$2"
a "A" b "$2"
```

## Attention Couple

Attention Couple is an attention-based implementation of regional prompting. it can often be faster and more flexible than latent-based masking.

The implementation is based on the one by [pamparamm](https://github.com/pamparamm/ComfyUI-ppm.git), but modified to use ComfyUI's hook system. This enables it to work with prompt scheduling.

The implementation produces slightly different results from Pamparamm's implementation because ComfyUI will only run the hook for conds that have it attached, unlike the ModelPatcher based implementation which has special logic to avoid messing up negative prompts with attention masks. It's also slightly slower because ComfyUI can't batch cond and uncond calculations while the hook is in use.

As a consequence of this, however, you can also use `ATTN()` in your negative prompt, and it will work correctly.

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

## TE_WEIGHT

For models using multiple text encoders, you can set weights per TE using the syntax `TE_WEIGHT(clipname=weight, clipname2=weight2, ...)` where `clipname` is one of the encoder names printed by `TE(help)`. For example with SDXL, try `TE_WEIGHT(g=0.25, l=0.75)`.

The weights are applied as a multiplier to the TE output. You can also override pooled output multipliers using eg. `l_pooled`.

To set a default value for all encoders, use `TE_WEIGHT(all=weight)`
