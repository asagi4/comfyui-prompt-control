# PC: Schedule Prompt

This node is the core of Prompt Control. It evaluates a prompt schedule and dynamically expands into a scheduled workflow consisting of calls to `PCTextEncode`, `SetConditioningTimesteps` and other necessary nodes.

To use it, simply replace your usual `CLIP Text Encode` nodes with `PC: Schedule Prompt` nodes. For LoRA Loading, you should use `PC: Schedule LoRAs` in place (or in addition to) of your usual LoRA Loader node.

For documentation on syntax, for now see the [documentation on GitHub](https://github.com/asagi4/comfyui-prompt-control/blob/master/doc/schedules.md)
