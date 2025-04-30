# ComfyUI prompt control

Control LoRA and prompt scheduling, advanced text encoding, regional prompting, and much more, through your text prompt. Generates dynamic graphs that are literally identical to handcrafted noodle soup.

A `Basic Text to Image` template is included with the extension, and can be loaded from ComfyUI's template library.

## Prompt Control v2

Prompt control has been almost completely rewritten. It now uses ComfyUI's lazy execution to build graphs from the text prompt at runtime. The generated graph is often exactly equivalent to a manually built workflow using native ComfyUI nodes. There are no more weird sampling hooks that could cause problems with other nodes

Prompt Control comes with `PCTextEncode`, which provides advanced text encoding with many additional features compared to ComfyUI's base `CLIPTextEncode`.

### Removed features

- Prompt interpolation syntax; it was too cumbersome to maintain
- LoRA block weight integration; ditto, for now.


### Everything broke, where are the old nodes?

If you really need them, you can install the [legacy nodes](https://github.com/asagi4/comfyui-prompt-control-legacy). However, I will not fix bugs in those nodes, and I strongly recommend just migrating your workflows to the new nodes.

You can have both installed at the same time; none of the nodes conflict.

## What can it do?

See [features](#features) below. Things you can control via the prompt:
- Prompt editing and filtering without noodle soup
- LoRA loading and scheduling via ComfyUI's hook system
- Masking, composition and area control (regional prompting)
- Prompt operations like `BREAK` and `AND`
- Weight interpretation types (comfy, A1111, etc.)
- Prompt masking with [cutoff](#cutoff)
- And a bunch more

See the [syntax documentation](doc/syntax.md)

If you find prompt scheduling inconvenient for some reason, `PCTextEncode` can be used as a drop-in replacement for `CLIPTextEncode` to get everything else.

[This workflow](example_workflows/Workflow%20Comparison.json?raw=1) shows LoRA scheduling and prompt editing and compares it with the same prompt implemented with built-in ComfyUI nodes. You can also find it in the template library.

The tools in this repository combine well with the macro and wildcard functionality in [comfyui-utility-nodes](https://github.com/asagi4/comfyui-utility-nodes)


## Requirements

For LoRA scheduling to work, you'll need at least version 0.3.7 of ComfyUI (0.3.36 of ComfyUI desktop).

You need to have `lark` installed in your Python environment for parsing to work (If you reuse A1111's venv, it'll already be there).

If you use the portable version of ComfyUI on Windows with its embedded Python, you must open a terminal in the ComfyUI installation directory and run the command:
```
.\python_embeded\python.exe -m pip install lark
```

Then restart ComfyUI afterwards.

# Core nodes

**Note**: The documentation refers to the nodes with their internal names for consistency. The display name may change, but ComfyUI's search will always find the nodes with the internal name. `PCLazyTextEncode` and `PCLazyLoraLoader` are the main ones you'll want to use, also known as `PC: Schedule Prompt` and `PC: Schedule LoRas`.

## PCLazyTextEncode and PCLazyTextEncodeAdvanced

`PCLazyTextEncode` uses ComfyUI's lazy graph execution mechanism to generate a graph of `PCTextEncode` and `SetConditioningTimestepRange` nodes from a prompt with schedules. This has the advantage that if a part of the schedule doesn't change, ComfyUI's caching mechanism allows you to avoid re-encoding the non-changed part.

for example, if you first encode `[cat:dog:0.1]` and later change that to `[cat:dog:0.5]`, no re-encoding takes place.

for added fun, put `NODE(NodeClassName, textinputname)` in a prompt to generate a graph using **any other node** that's compatible. The node can't have required parameters besides a single CLIP parameter (which must be named `clip`) and the text prompt, and it must return a `CONDITIONING` as its first return value. The "default" values are `PCTextEncode` and `text`.

For example, if you for some reason do not want the advanced features of `PCTextEncode`, use `NODE(CLIPTextEncode)` in the prompt and you'll still get scheduling with ComfyUI's regular TE node.

The advanced node enables filtering the prompt for multi-pass workflows.

## PCLazyLoraLoader and PCLazyLoraLoaderAdvanced

This node reads LoRA expressions from the scheduled prompt and constructs a graph of `LoraLoader`s and `CreateHookLora`s as necessary to provide the necessary LoRA scheduling. Just use it in place of a `LoRALoader` and use the output normally.

The Advanced node gives you access to the generated hooks. If you have `apply_hooks` set to true, you **do not** need to apply the `HOOKS`  output to a CLIP model separately; it's provided in case you want to use it elsewhere. The advanced node also enables filtering the prompt for multi-pass workflows.

## PCTextEncode

Encodes a single prompt with advanced (non-scheduling) syntax enabled. This is what actually does most of the work under the hood.

Note: `PCTextEncode` **does not** ignore `<lora:...:1>` and will treat it as part of the prompt. To use a combined prompt for LoRAs and your input, use `PCLazyTextEncode` and `PCLazyLoraLoader`

## PCAddMaskToCLIP

This node attaches masks to a `CLIP` model so that they can be referred to when using the `IMASK` custom mask function of `PCTextEncode`.

## PCSetTextEncodeSettings

This node configures `PCTextEncode` default values for some functions by attaching the information to a `CLIP` model.

# Features
## Scheduling and LoRA loading

Prompt control provides a way to easily schedule different prompts and control LoRA loading.

See the [syntax documentation](doc/syntax.md)

### Note on how schedules work

ComfyUI does not use the step number to determine whether to apply conds; instead, it uses the sampler's timestep value which is affected by the scheduler you're using. This means that when the sampler scheduler isn't linear, the schedules generated by prompt control will not be either.

## Advanced CLIP encoding

If you use `PCTextEncode`, advanced encodings are available automatically. Thanks to BlenderNeko for the original code.

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

For things (ie. the code imports) to work, the nodes must be cloned in a directory named exactly `ComfyUI_ADV_CLIP_emb`.

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

# Known issues

- ComfyUI's caching mechanism has an issue that makes it unnecessarily invalidate caches for certain inputs; you'll still get some benefit from the lazy nodes, but changing inputs that shouldn't affect downstream nodes (especially if using filtering) will still cause them to be recomputed because ComfyUI doesn't realize the inputs haven't changed.

If you want to enable a hack to fix this, set `PROMPTCONTROL_ENABLE_CACHE_HACK=1` in your environment. Unset it to disable.

It's a purely optional performance optimization that allows Prompt Control nodes to override their cache keys in a way that should not interfere with other nodes. Note that the optimization only works if the text input to the lazy nodes is a constant (so either directly on the node or from a primitive); outputs from other nodes can't be optimized.
