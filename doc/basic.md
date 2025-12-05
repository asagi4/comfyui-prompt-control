# Basic Prompt Syntax

The syntax below documents the features of `PCTextEncode`

## Combining prompts

### AND

`AND` can be used to create "prompt segments". By default, it works as if you had combined the different prompts with `ConditioningCombine`. 

It is also used with regional prompting, see `MASK` and `COUPLE` below.

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

If `AND` is placed inside quotes (eg. `Text saying "CAT AND DOG"`) it will be treated as regular text.

## Note about processing order

Prompt operators are processed in the following order, meaning that all features "below" another can be affected by the feature above it. That is, `BREAK` can go inside a `TE()` call, but not `AND` or `CAT`.

- DEF macros are expanded
- Scheduling is expanded, and for each scheduled prompt:
  - The prompt is split by AND, and for each:
    - Prompts are split by COUPLE. and for each:
      - Most functions (like MASK) and cutoffs are evaluated
      - prompts are split by `AVG()` or CAT
        - the TE() function is evaluated to set per-encoder prompts
        - BREAK is evaluated
        - Everything else
      - Prompts are combined with `ConditioningAverage` (for `AVG`) or `ConditioningConcat` (for `CAT`)
    - If coupled prompts exist, the base cond is set up for attention coupling and returned
  - Prompts split with `AND` are combined with `ConditioningCombine`
- Each scheduled prompt is restricted to its effective range with  `ConditioningSetTimestepRange`

## Functions

There are some "functions" that can be included in a prompt to affect how it is interpreted.

Functions have the form `FUNCNAME(param1, param2, ...)`. How parameters are interpreted is up to the function. 

In general, function parameters will have default values that are used if the parameter is left empty.

Note: Whitespace is usually *not* stripped from string parameters by default. Commas can be escaped with `\,`

Like `AND`, functions are parsed after regular scheduling syntax has been expanded, allowing things like `[AREA:MASK:0.3](...)`, in case that's somehow useful.

like AND, if any function is placed inside quotes, it will *not* activate and is instead treated as regular text.

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

See [Regional prompting](/doc/regional_prompting.md)

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

> [!WARN]
> These features are may change or disappear without warning

## COUPLE: Attention couple

See [here](/doc/attention_couple.md)

## TE_WEIGHT

For models using multiple text encoders, you can set weights per TE using the syntax `TE_WEIGHT(clipname=weight, clipname2=weight2, ...)` where `clipname` is one of the encoder names printed by `TE(help)`. For example with SDXL, try `TE_WEIGHT(g=0.25, l=0.75)`.

The weights are applied as a multiplier to the TE output. You can also override pooled output multipliers using eg. `l_pooled`.

To set a default value for all encoders, use `TE_WEIGHT(all=weight)`
