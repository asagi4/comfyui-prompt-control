# ComfyUI prompt control

Control LoRA and prompt scheduling, advanced text encoding, regional prompting, and much more, through your text prompt. Generates dynamic graphs that are literally identical to handcrafted noodle soup.

Prompt Control comes with `PCTextEncode`, which provides advanced text encoding with many additional features compared to ComfyUI's base `CLIPTextEncode`.

A `Basic Text to Image` template is included with the extension, and can be loaded from ComfyUI's template library.

## What can it do?

You can use text prompts to control the following:

- A1111-style prompt scheduling and filtering without noodle soup.
- LoRA loading and [scheduling](/doc/schedules.md) via the prompt, using ComfyUI's hook system
- Masking, composition and area control ([regional prompting](/doc/regional_prompts.md)) with an implementation of [Attention Couple](/doc/attention_couple.md), also fully schedulable.
- [Advanced prompt encoding](/doc/basic.md)
  - Per-encoder prompts for models with multiple text encoders, such as SDXL and Flux
  - Prompt combinators like `BREAK`, as well as `CAT`, `AVG()` and `AND` corresponding to ComfyUI's `ConditioningConcat`, `ConditioningAverage` and `ConditioningCombine` nodes.
  - Different weight interpretation types (ComfyUI, A1111, compel, etc.)
  - Prompt masking with an implementation of [cutoff](https://github.com/BlenderNeko/ComfyUI_Cutoff)
- Simple [prompt macros](/doc/macros.md) with `DEF`

All features are fully schedulable unless otherwise stated. See the [scheduling syntax documentation](doc/schedules.md) to get started.

If you find prompt scheduling inconvenient for some reason, `PCTextEncode` can be used as a drop-in replacement for `CLIPTextEncode` to get everything else.

[This workflow](example_workflows/Workflow%20Comparison.json?raw=1) shows LoRA scheduling and prompt editing and compares it with the same prompt implemented with built-in ComfyUI nodes. You can also find it in the template library.

## Compatibility

Prompt Control uses graph generation,  and tries to delegate functionality to core ComfyUI wherever possible, implementing any hooks and patches in a way that is maximally compatible. This means that it should just work in most cases, even with models and nodes not explicitly supported.

If you encounter issues as a user or if you're a node developer and Prompt Control somehow breaks something, feel free to file a bug report.

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

- Cutoff does not work with models that use non-CLIP text encoders, like Flux. This might be fixable, but it's uncertain if cutoff even makes sense for those models.
