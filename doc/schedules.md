# Prompt Schedule Syntax

> [!TIP]
> If you're viewing this on GitHub, I recommend opening the outline by clicking the button in the top right corner of the text view (it is annoyingly easy to miss).

> [!NOTE]
> The syntax documented in this section is only available with the `PC: Schedule Prompt` and `PC: Schedule LoRAs` nodes and their advanced variants.

Scheduling syntax is available with is similar to A1111, but only fractions are supported for steps. LoRAs are scheduled by including them in a scheduling expression.

Besides the syntax documented below, the [basic syntax](/doc/basic.md) and [prompt macro](/doc/macros.md) features are also automatically available.

```
a [large::0.1] [cat|dog:0.05] [<lora:somelora:0.5:0.6>::0.5]
[in a park:in space:0.4]
```
## Comments and escaping

In schedules, any text on a line following a `#` is considered a comment and removed, including the `#` character.
You can escape the following characters in places where they would otherwise conflict with syntax:

- `#` with `\#`
- `:` with `\:`
- `\` with `\\`

Escaping is only required if it would otherwise be considered syntax, that is `\o/` will be interpreted literally and the `\` does not need to be escaped, but in `[embedding:a:0.5]` you would need to escape the `:`.

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

The most general form of a schedule is a range expression: For example, in `prompt [before:during:after:0.3,0.7]`, The prompt be `prompt before` until 0.3, `prompt during` until 0.7, and then `prompt after`. This form is equivalent to `prompt [before:[during:after:0.7]:0.3]`

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

You can also give the exact path (including the extension) as shown in `LoRALoader`.

If no match is found, the node will try to replace spaces with underscores and search again. That is, `<lora:cats and dogs:1>` will find `cats_and_dogs.safetensors`. This helps with some autocompletion scripts that replace underscores with spaces.

Finally, if none of the above produce a match, the search term will be split by whitespace and files that contain all of the parts in any order will be considered. If this returns only a single match, it will be loaded. For example, consider LoRAs:

- `xl/red_cats.safetensors`
- `flux/blue_cats.safetensors`
- `flux/red_cats.safetensors`

Then `<lora:cats xl:1>` would match the red cats LoRA, but `cats flux` would be ambiguous and not match.

## Alternating

Alternating syntax is `[a|b:pct_steps]`, causing the prompt to alternate every `pct_steps`. `pct_steps` defaults to 0.1 if not specified. You can also have more than two options.


## Sequences

The syntax `[SEQ:a:N1:b:N2:c:N3]` is shorthand for `[a:[b:[c::N3]:N2]:N1]` ie. it switches from `a` to `b` to `c` to nothing at the specified points in sequence.

Might be useful with Jinja templating (see https://github.com/asagi4/comfyui-utility-nodes). For example:
```
[SEQ<% for x in steps(0.1, 0.9, 0.1) %>:<lora:test:<= sin(x*pi) + 0.1 =>>:<= x =><% endfor %>]
```
generates a LoRA schedule based on a sinewave
