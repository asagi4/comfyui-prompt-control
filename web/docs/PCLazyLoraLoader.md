# PC: Schedule LoRAs

This node is the core of Prompt Control. It evaluates a prompt schedule and dynamically expands into a scheduled workflow consisting of necessary calls to `LoRALoader` and `Create Hook LoRA` (for scheduled LoRAs).

You can use it in place or in addition to your usual `LoRA Loader` nodes; just pass in a text prompt containing your LoRA schedule (it can be shared with `PC: Schedule Prompt`). Then connect your MODEL output as usual and the CLIP output to your `PC: Schedule Prompt` nodes.

For documentation on syntax, for now see the [documentation on GitHub](https://github.com/asagi4/comfyui-prompt-control/blob/master/doc/schedules.md)
