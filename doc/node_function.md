# The NODE function

The `NODE` function allows you to use any other text encoding node within `PC: Schedule Prompt`, replacing the default `PCTextEncode` and allowing for example video model scheduling.

> [!NOTE]
> When using NODE, you lose access to *all* special syntax provided by `PCTextEncode`. Only SEGs, macros and scheduling will continue to work since those are processed at graph expansion time before the text prompt is passed into the node.

## Basic usage

Use `NODE(NodeClassName, textinputname)` in a prompt to generate a graph using any node that's compatible. The requirements are as follows:
- The node must have a CLIP parameter (which must be named `clip`)
- It must have a text field
- It must return a `CONDITIONING` as its first return value.

For example, if you for some reason do not want the advanced features of `PCTextEncode`, use `NODE(CLIPTextEncode)` in the prompt and you'll still get scheduling with ComfyUI's regular TE node.

The default parameters are `PCTextEncode` and `text`.

## Advanced Usage with arbitrary parameters

Advanced usage of `NODE` can be complicated. For an example, see [The H3 workflow](/example_workflows/Prompt%20Control%20with%20MiniMax%20H3.json?raw=1). You can also find it in the template library.

The full synopsis of the function is `NODE(NodeClassName, textinputname, arg_spec)` where `arg_spec` is a semicolon-separated list of `parameter_name json_value` pairs. In raw form, it looks like this:
```
NODE(MiniMaxH3ImageToVideo, prompt, vae ["1", 0]; width 1024; height 1024; first_frame ["2", 0])
```

The names and inputs must match the ComfyUI API format which **may differ from frontend names**. You can export your workflow in API format and inspect it to see how inputs are passed in to nodes.

The arrays are literal ComfyUI node links, meaning the `vae` parameter is taken from node ID "1" first output and `first_frame` from node ID "2" first output.

The values are arbitrary JSON literals, meaning that you can also pass in constant values. To pass in literal strings for example, you need to use quotes `"like this"`.

This is intended to be used with the helper node `PC: NODE Input Helper`, which can be used to pass arbitrary parameters (named `$a` to `$n`) to the encoder. The recommended pattern is to put something like the following:
```
SEG(node)
NODE(MiniMaxH3ImageToVideo, prompt, vae $a; width $b; height $c; length $d; first_frame $e)
```
to the helper and then concatenate it at the end of your prompt (use whitespace as a separator). You can then trigger the node with `$node` in your prompt. (see [documentation](/doc/macros.md) for `SEG`)

The helper will replace the parameters with the correct ComfyUI link values.
