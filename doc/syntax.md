# Scheduling syntax

Syntax is like A1111 for now, but only fractions are supported for steps. LoRAs are scheduled by including them in a scheduling expression.

```
a [large::0.1] [cat|dog:0.05] [<lora:somelora:0.5:0.6>::0.5]
[in a park:in space:0.4]
```

## Scheduled prompts

There are two forms of scheduled prompts.

### Basic scheduling expressions
Basic expressions take the form `[before:after:X]` where `X` is the switch point, a decimal number between 0.0 and 1.0 inclusive, representing 0 to 100% of timesteps.
For example:
```
a [red:blue:0.5] cat
```
switches from `a red cat` to `a blue cat` at 0.5. `before` and `after` can be arbitrary prompts (`after` can also be empty), including other scheduling expressions, allowing nesting:
```
a [red:[blue::0.7]:0.5] cat
```

switches from `a red cat` to `a blue cat` at 0.5 and to `a cat` at 0.7

**Note:** As a special case, `[cat:0.5]` is equivalent to `[:cat:0.5]` meaning it switches from empty to `cat` at 0.5.

### Using step numbers with the Advanced nodes

If you provide a non-zero value to `num_steps` to the `Advanced` versions of the scheduling nodes, you will be able to use step numbers in prompts.

For now, a value between 0 and 1.0 will be interpreted as a percentage if it contains a ., and as an absolute step otherwise.

This is just syntactic sugar. Behind the scenes, the values are converted to percentages and have normal ComfyUI scheduling behaviour.

### Range expressions

You can also use `a [during:after:0.3,0.7]` as a shortcut. The prompt be `a` until 0.3, `a during` until 0.7, and then `a after`. This form is equivalent to `[[during:after:0.7]:0.3]`

For convenience, `[during:0.1,0.4]` is equivalent to `[during::0.1,0.4]`

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

This syntax is also available in outside scheduled prompts, where applicable.

## LoRA loading

The A111-style syntax `<lora:loraname:weight>` can be used to load LoRAs via the prompt. See LoRA scheduling above.

## Combining prompts, A1111-style

- The keyword `BREAK` causes the prompt to be tokenized in separate chunks, which results in each chunk being individually padded to the text encoder's maximum token length. This is mostly equivalent to the `ConditioningConcat` node.

`AND` can be used to combine prompts. You can also use a weight at the end. It does a weighted sum of each prompt,
```
cat :1 AND dog :2
```
The weight defaults to 1 and are normalized so that `a:2 AND b:2` is equal to `a AND b`. `AND` is processed after schedule parsing, so you can change the weight mid-prompt: `cat:[1:2:0.5] AND dog`

## Functions

There are some "functions" that can be included in a prompt to do various things. 

Functions have the form `FUNCNAME(param1, param2, ...)`. How parameters are interpreted is up to the function.
Note: Whitespace is *not* stripped from string parameters by default. Commas can be escaped with `\,`

Like `AND`, these functions are parsed after regular scheduling syntax has been expanded, allowing things like `[AREA:MASK:0.3](...)`, in case that's somehow useful.

### SDXL

The nodes do not treat SDXL models specially, but there are some utilities that enable SDXL specific functionality.

You can use the function `SDXL(width height, target_width target_height, crop_w crop_h)` to set SDXL prompt parameters. `SDXL()` is equivalent to `SDXL(1024 1024, 1024 1024, 0 0)` unless the default values have been overridden by `PCScheduleSettings`.

### Multiple text encoders: TE

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
- `AND` inside `TE` does not do anything sensible; `TE(l=foo AND bar)` will parse as two prompts `TE(foo` and `bar)`. `BREAK`, `SHIFT` and `SHUFFLE` do work, however

### SHUFFLE and SHIFT

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

### NOISE

The function `NOISE(weight, seed)` adds some random noise into the prompt. The seed is optional, and if not specified, the global RNG is used. `weight` should be between 0 and 1.

### MASK, IMASK and AREA

You can use `MASK(x1 x2, y1 y2, weight, op)` to specify a region mask for a prompt. The values are specified as a percentage with a float between `0` and `1`, or as absolute pixel values (these can't be mixed). `1` will be interpreted as a percentage instead of a pixel value.

Multiple `MASK` or `IMASK` calls will be composited together using ComfyUI's `MaskComposite` node, using `op` as the `operation` parameter (defaulting to `multiply`).

Similarly, you can use `AREA(x1 x2, y1 y2, weight)` to specify an area for the prompt (see ComfyUI's area composition examples). The area is calculated by ComfyUI relative to your latent size.

#### Custom masks: IMASK and `PCAddMaskToCLIP`

You can attach custom masks to a `CLIP` with the `PC: Attach Mask` nodes and then refer to those masks in the prompt using `IMASK(index, weight, op)`. Indexing starts from zero, so 0 is the first attached mask etc. `PCSCheduleAddMasks` ignores empty inputs, so if you only add a mask to the `mask4` input, it will still have index 0.

Applying the nodes multiple times *appends* masks rather than overriding existing ones, so if you need more than 4, you can just use it more than once.

#### Behaviour of masks
If multiple `MASK`s are specified, they are combined together with ComfyUI's `MaskComposite` node, with `op` specifying the operation to use (default `multiply`). In this case, the combined mask weight can be set with `MASKW(weight)` (defaults to 1.0).

Masks assume a size of `(512, 512)`, unless overridden with `PC: Configure PCTextEncode` and pixel values will be relative to that. ComfyUI will scale the mask to match the image resolution. You can change it manually by using `MASK_SIZE(width, height)` anywhere in the prompt,

These are handled per `AND`-ed prompt, so in `prompt1 AND MASK(...) prompt2`, the mask will only affect prompt2.

The default values are `MASK(0 1, 0 1, 1)` and you can omit unnecessary ones, that is, `MASK(0 0.5, 0.3)` is `MASK(0 0.5, 0.3 1, 1)`

Note that because the default values are percentages, `MASK(0 256, 64 512)` is valid, but `MASK(0 200)` will raise an error.

Masking does not affect LoRA scheduling unless you set unet weights to 0 for a LoRA.

### FEATHER

When you use `MASK` or `IMASK`, you can also call `FEATHER(left top right bottom)` to apply feathering using ComfyUI's `FeatherMask` node. The values are in pixels and default to `0`.

If multiple masks are used, `FEATHER` is applied *before compositing* in the order they appear in the prompt, and any leftovers are applied to the combined mask. If you want to skip feathering a mask while compositing, just use `FEATHER()` with no arguments.

For example:
```
MASK(1) MASK(2) MASK(3) FEATHER(1) FEATHER() FEATHER(3) weirdmask FEATHER(4)
```

gives you a mask that is a combination of 1, 2 and 3, where 1 and 3 are feathered before compositing and then `FEATHER(4)` is applied to the composite.

The order of the `FEATHER` and `MASK` calls doesn't matter; you can have `FEATHER` before `MASK` or even interleave them.

## Miscellaneous
- `<emb:xyz>` is alternative syntax for `embedding:xyz` to work around a syntax conflict with `[embedding:xyz:0.5]` which is parsed as a schedule that switches from `embedding` to `xyz`.

# Experimental features

Experimental features are unstable and may disappear or change without warning.

## DEF

You can define "prompt macros" by using `DEF`:
```
DEF(MYMACRO=this is a prompt)
[(MYMACRO:0.6):(MYMACRO:1.1):0.5]
```
is equivalent to
```
[(this is a prompt:0.5):(this is a prompt:1.1):0.5]
```

It's also possible to give parameters to a macro:
```
DEF(MYMACRO=[(prompt $1:$2):(prompt $1:$3):$4])
MYMACRO(test; 1.1; 0.7; 0.2)
```
gives
```
[(prompt test:1.1):(prompt test:0.7):0.2]
```
in this form, the variables $N (where N is any number corresponding to a positional parameter) will be replaced with the given parameter. The parameters must be separated with a semicolon.

You can also optionally specify default values:

```
DEF(MACRO(example; 0; 1)=[$1:$2,$3])
MACRO MACRO(test; 0.2)
```
gives
```
[example:0,1] [test:0.2,1]
```

Note that unspecified parameters will not be substituted:
```
DEF(mything=a $1 b $2)
mything
mything(A)
```
gives
```
a $1 b $2
a A b $2
```

Macros are expanded before any other parsing takes place. The expansion continues until no further changes occur. Recursion will raise an error.

## Attention masking

Use `ATTN()` in combination with `MASK()` or `IMASK()` to enable attention masking. Currently, it's pretty slow and only works with SDXL. You need to have a recent enough version of ComfyUI for this to work.

## TE_WEIGHT

For models using multiple text encoders, you can set weights per TE using the syntax `TE_WEIGHT(clipname=weight, clipname2=weight2, ...)` where `clipname` is one of the encoder names printed by `TE(help)`. For example with SDXL, try `TE_WEIGHT(g=0.25, l=0.75)`.

The weights are applied as a multiplier to the TE output. You can also override pooled output multipliers using eg. `l_pooled`.

To set a default value for all encoders, use `TE_WEIGHT(all=weight)`
