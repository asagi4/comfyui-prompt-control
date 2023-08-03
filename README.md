# ComfyUI prompt control

Nodes for convenient prompt editing. The aim is to make basic generations in ComfyUI completely prompt-controllable.

You need to have `lark` installed in your Python environment for parsing to work (If you reuse A1111's venv, it'll already be there)

The basic nodes should now be stable, though I won't make interface guarantees quite yet.

[This example workflow](workflows/example.json) implements a two-pass workflow illustrating most features.

## Syntax

Syntax is like A1111 for now, but only fractions are supported for steps.

```
a [large::0.1] [cat|dog:0.05] [<lora:somelora:0.5:0.6>::0.5]
[in a park:in space:0.4]
```
### Alternating

Alternating syntax is `[a|b:pct_steps]`, causing the prompt to alternate every `pct_steps`. `pct_steps` defaults to 0.1 if not specified. You can also have more than two options.

### Sequences

The syntax `[SEQ:a:N1:b:N2:c:N3]` is shorthand for `[a:[b:[c::N3]:N2]:N1]` ie. it switches from `a` to `b` to `c` to nothing at the specified points in sequence.

Might be useful with Jinja templating (see Experiments below for details). For example:
```
[SEQ<% for x in steps(0.1, 0.9, 0.1) %>:<lora:test:<= sin(x*pi) + 0.1 =>>:<= x =><% endfor %>]
```

generates a LoRA schedule based on a sinewave

### Tag selection
Instead of step percentages, you can use a *tag* to select part of an input:
```
a large [dog:cat<lora:catlora:0.5>:SECOND_PASS]
```
You can then use the `tags` parameter in the `FilterSchedule` node to filter the prompt. If the tag matches any tag `tags` (comma-separated), the second option is returned (`cat`, in this case, with the LoRA). Otherwise, the first option is chosen (`dog`, without LoRA).

the values in `tags` are case-insensitive, but the tags in the input **must** be uppercase A-Z and underscores only, or they won't be recognized. That is, `[dog:cat:hr]` will not work.

For example, a prompt
```
a [black:blue:X] [cat:dog:Y] [walking:running:Z] in space
```
with `tags` `x,z` would result in the prompt `a blue cat running in space`

## Schedulable LoRAs
The `ScheduleToModel` node patches a model such that when sampling, it'll switch LoRAs between steps. You can apply the LoRA's effect separately to CLIP conditioning and the unet (model)

For me this seems to be quite slow without the --highvram switch because ComfyUI will shuffle things between the CPU and GPU. YMMV. When things stay on the GPU, it's quite fast.

## AITemplate support
LoRA scheduling supports AITemplate.

Due to sampler patching, your AITemplate nodes must be cloned to a directory called `AIT` under `custom_nodes` or the hijack won't find it.

Note that feeding too large conditionings to AITemplate seems to break it. This can happen when using alternating syntax with too small a step.

## Nodes

### PromptToSchedule
Parses a schedule from a text prompt. A schedule is essentially an array of `(valid_until, prompt)` pairs that the other nodes can use.

### FilterSchedule
Filters a schedule according to its parameters, removing any *changes* that do not occur within `[start, end)` as well as doing tag filtering. Always returns at least the last prompt in the schedule if everything would otherwise be filtered, so `start=1.0, end=1.0` returns the prompt at 1.0.

### ScheduleToCond
Produces a combined conditioning for the appropriate timesteps. From a schedule. Also applies LoRAs to the CLIP model according to the schedule.

### ScheduleToModel
Produces a model that'll cause the sampler to reapply LoRAs at specific steps according to the schedule.

This depends on a callback handled by a monkeypatch of the ComfyUI sampler function, so it might not work with custom samplers, but it shouldn't interfere with them either.

### JinjaRender
Renders a String with Jinja2. See below for details

## Older nodes

- `EditableCLIPEncode`: A combination of `PromptToSchedule` and `ScheduleToCond`
- `LoRAScheduler`: A combination of `PromptToSchedule`, `FilterSchedule` and `ScheduleToModel`

## Utility nodes
### StringConcat
Concatenates the input strings into one string. All inputs default to the empty string if not specified

### ConditioningCutoff
Removes conditionings from the input whose timestep range ends before the cutoff and extends the remaining conds to cover the missing part. For example, set the cutoff to 1.0 to only leave the last prompt. This can be useful for HR passes.

# Experiments

## Jinja2
You can use the `JinjaRender` node to evaluate a string as a Jinja2 template. Note, however, that because ComfyUI's frontend uses `{}` for syntax, There are the following modifications to Jinja syntax:

- `{% %}` becomes `<% %>`
- `{{ }}` becomes `<= =>`
- `{# #}` becomes `<# #>`

Jinja stuff is experimental.

### Functions in Jinja templates

The following functions and constants are available:

- `pi`
- `min`, `max`, `clamp(minimum, value, maximum)`,
- `abs`, `round`, `ceil`, `floor`
- `sqrt` `sin`, `cos`, `tan`, `asin`, `acos`, `atan`. These functions are rounded to two decimals


In addition, a special `steps` function exists.

The `steps` function will generate a list of steps for iterating. 

You can call it either as `steps(end)`, `steps(end, step=0.1)` or `steps(start, end, step)`. `step` is an optional parameter that defaults to `0.1`. It'll return steps *inclusive* of start and end as long as step doesn't go past the end. 

The second form is equivalent to `steps(step, end, step)`. i.e. it starts at the first step.

# TODO & BUGS

The loaders can mostly reproduce the output from using `LoraLoader`.

There are some very minor differences compared to using multiple sampling passes when LoRA scheduling is in use. Changing this would require changing the implementation to call sampling multiple times. I may do this at some point, but it's good enough for now.

More advanced workflows might explode horribly.

- If execution is interrupted and LoRA scheduling is used, your models might be left in an undefined state until you restart ComfyUI
- Needs better syntax. A1111 is familiar, but not very good
- More advanced LoRA weight scheduling
