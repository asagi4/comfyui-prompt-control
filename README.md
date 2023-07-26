# ComfyUI prompt control

Nodes for convenient prompt editing. The aim is to make basic generations in ComfyUI completely prompt-controllable.

*very* experimental, but things seem to work okay.

Syntax is like A1111 for now, but only fractions are supported for steps:

LoRAs work too, composable-lora -style

```
a [large::0.1] [cat:dog:0.5] [<lora:somelora:0.5:0.6>::0.5]
```
The `example.json` contains a simple workflow to play around with.

## Schedulable LoRAs
The `LoRAScheduler` node patches a model such that when sampling, it'll switch LoRAs between steps. You can apply the LoRA's effect separately to CLIP conditioning and the unet (model)

For me this seems to be quite slow without the --highvram switch because ComfyUI will shuffle things between the CPU and GPU. YMMV. When things stay on the GPU, it's quite fast.


# TODO & BUGS

The basics seem to work; without prompt editing, the loaders can reproduce the output from using `LoraLoader`

More advanced workflows might explode horribly.

- Re-add sampler node with scheduling support and see if it produces different results to using an array of conds
- better syntax. A1111 is familiar, but not very good
- alternating
- convenient prompt editing for multiple sampling passes (HR fix etc)
