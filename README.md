# ComfyUI prompt control

Nodes for convenient prompt editing. The aim is to make basic generations in ComfyUI completely prompt-controllable.

You need to have `lark` installed in your Python envinronment for parsing to work (If you reuse A1111's venv, it'll already be there)

*very* experimental, but things seem to work okay.

LoRAs work too, composable-lora -style

## Syntax

Syntax is like A1111 for now, but only fractions are supported for steps.

```
a [large::0.1] [cat|dog:0.05] [<lora:somelora:0.5:0.6>::0.5]
[in a park:in space:0.4]
```
### Alternating

Alternating syntax is `[a|b:pct_steps]`, causing the prompt to alternate every `pct_steps`. `pct_steps` defaults to 0.1 if not specified.
The `example.json` contains a simple workflow to play around with.

### Sequences

The syntax `[SEQ:a:N1:b:N2:c:N3]` is shorthand for `[a:[b:[c::N3]:N2]:N1]` ie. it switches from `a` to `b` to `c` to nothing at the specified points in sequence.

Might be useful with Jinja templating (see Experiments below for details). For example:
```
[SEQ<% for x in steps(0.1, 0.9, 0.1) %>:<lora:test:<= m.sin(x*m.pi) + 0.1 =>>:<= x =><% endfor %>]
```

generates a LoRA schedule based on a sinewave

### Tag selection
Instead of step percentages, you can use a *tag* to select part of an input:
```
a large [dog:cat<lora:catlora:0.5>:SECOND_PASS]
```
then specify the `filter_tags` parameter in `LoRAScheduler` or `EditableCLIPEncoding` to filter the prompt. If the tag matches any tag `filter_tags` (comma-separated), the second option is returned (`cat`, in this case, with the LoRA). Otherwise, the first option is chosen (`dog`, without LoRA).

the values in `filter_tags` are case-insensitive, but the tags in the input **must** be uppercase A-Z and underscores only, or they won't be recognized. That is, `[dog:cat:hr]` will not work.

For example, a prompt
```
a [black:blue:X] [cat:dog:Y] [walking:running:Z] in space
```
with `filter_tags` `x,z` would result in the prompt `a blue cat running in space`

## Nodes

### EditableCLIPEncode
Parses a prompt and produces a combined conditioning for the appropriate timesteps. Also applies LoRAs to the CLIP model according to the prompt.

You need a recent enough version of ComfyUI to support timestep ranges.

### LoRAScheduler
Parses a prompt and produces a model that'll cause the sampler to reapply LoRAs at specific steps.

This depends on a callback handled by a monkeypatch of the ComfyUI sampler function, so it might not work with custom samplers, but it shouldn't interfere with them either.

`cutoff` works like `ConditioningCutoff` below.

### ConditioningCutoff
Removes conditionings from the input whose timestep range ends before the cutoff and extends the remaining conds to cover the missing part. For example, set the cutoff to 1.0 to only leave the last prompt. This can be useful for HR passes.

## Schedulable LoRAs
The `LoRAScheduler` node patches a model such that when sampling, it'll switch LoRAs between steps. You can apply the LoRA's effect separately to CLIP conditioning and the unet (model)

For me this seems to be quite slow without the --highvram switch because ComfyUI will shuffle things between the CPU and GPU. YMMV. When things stay on the GPU, it's quite fast.

## AITemplate support
LoRA scheduling supports AITemplate. 

Due to sampler patching, your AITemplate nodes must be cloned to a directory called `AIT` under `custom_nodes` or the hijack won't find it.

Note that feeding too large conditionings to AITemplate seems to break it. This can happen when using alternating syntax with too small a step.

# Experiments

If you have the `jinja2` package installed in your Python environment, your prompts will be evaluated as a Jinja2 template prior to parsing. Note, however, that because ComfyUI's frontend uses `{}` for syntax, There are the following modifications to Jinja syntax:

- `{% %}` is `<% %>`
- `{{ }}` is `<= =>`
- `{# #}` is `<# #>`

Jinja stuff is experimental. I might make it its own node at some point instead of stuffing everything in the same parser.

The Python `math` module is available to jinja as the variable `m`. See `import math; help(math)` in a Python interpreter.

There's also a handy `steps` function that'll generate a list of steps for iterating.

You can call it either as `steps(end)`, `steps(end, step=0.1)` or `steps(start, end, step)`. `step` is an optional parameter that defaults to `0.1`. It'll return steps *inclusive* of start and end as long as step doesn't go past the end.

the second form is equivalent to `steps(step, end, step)`. i.e. it starts at the first step.

# TODO & BUGS

The loaders can mostly reproduce the output from using `LoraLoader`.

There are some very minor differences compared to using multiple sampling passes when LoRA scheduling is in use. Changing this would require changing the implementation to call sampling multiple times. I may do this at some point, but it's good enough for now.

More advanced workflows might explode horribly.

- If execution is interrupted and LoRA scheduling is used, your models might be left in an undefined state until you restart ComfyUI
- Needs better syntax. A1111 is familiar, but not very good
- More advanced LoRA weight scheduling
