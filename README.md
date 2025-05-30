# ComfyUI prompt control

Control LoRA and prompt scheduling, advanced text encoding, regional prompting, and much more, through your text prompt. Generates dynamic graphs that are literally identical to handcrafted noodle soup.

A `Basic Text to Image` template is included with the extension, and can be loaded from ComfyUI's template library.

## What can it do?

You can use text prompts to control the following:

- Prompt scheduling and filtering without noodle soup.
- LoRA loading and scheduling via ComfyUI's hook system
- Masking, composition and area control (regional prompting) with an implementation of Attention Couple, also fully schedulable.
- Per-encoder prompts for models with multiple text encoders, such as SDXL and Flux
- Prompt operations like `BREAK` and `AND`
- Different weight interpretation types (ComfyUI, A1111, compel, etc.)
- Prompt masking with an implementation of [cutoff](https://github.com/BlenderNeko/ComfyUI_Cutoff)
- Simple prompt macros with `DEF`
- And a bunch more

All features are fully schedulable unless otherwise stated. See the [syntax documentation](doc/syntax.md) for details on how to use each feature.

If you find prompt scheduling inconvenient for some reason, `PCTextEncode` can be used as a drop-in replacement for `CLIPTextEncode` to get everything else.

[This workflow](example_workflows/Workflow%20Comparison.json?raw=1) shows LoRA scheduling and prompt editing and compares it with the same prompt implemented with built-in ComfyUI nodes. You can also find it in the template library.

## Prompt Control v2

Prompt control has been almost completely rewritten. It now uses ComfyUI's lazy execution to build graphs from the text prompt at runtime. The generated graph is often exactly equivalent to a manually built workflow using native ComfyUI nodes. There are no more weird sampling hooks that could cause problems with other nodes

Prompt Control comes with `PCTextEncode`, which provides advanced text encoding with many additional features compared to ComfyUI's base `CLIPTextEncode`.

### Removed features

- Prompt interpolation syntax; it was too cumbersome to maintain
- LoRA block weight integration; ditto, for now.

### Everything broke, where are the old nodes?

If you really need them, you can install the [legacy nodes](https://github.com/asagi4/comfyui-prompt-control-legacy). However, I will not fix bugs in those nodes, and I strongly recommend just migrating your workflows to the new nodes.

You can have both installed at the same time; none of the nodes conflict.

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

# Known issues

- ComfyUI's caching mechanism has an issue that makes it unnecessarily invalidate caches for certain inputs; you'll still get some benefit from the lazy nodes, but changing inputs that shouldn't affect downstream nodes (especially if using filtering) will still cause them to be recomputed because ComfyUI doesn't realize the inputs haven't changed.

If you want to enable a hack to fix this, set `PROMPTCONTROL_ENABLE_CACHE_HACK=1` in your environment. Unset it to disable.

It's a purely optional performance optimization that allows Prompt Control nodes to override their cache keys in a way that should not interfere with other nodes. Note that the optimization only works if the text input to the lazy nodes is a constant (so either directly on the node or from a primitive); outputs from other nodes can't be optimized.
