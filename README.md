# ComfyUI prompt control

Nodes for convenient prompt editing. The aim is to make basic generations in ComfyUI completely prompt-controllable.

*very* experimental, but things seem to work okay.

LoRAs work too, composable-lora -style
Syntax is like A1111 for now, but only fractions are supported for steps.

Alternating syntax is `[a|b:pct_steps]`, causing the prompt to alternate every `pct_steps`. `pct_steps` defaults to 0.1 if not specified.

```
a [large::0.1] [cat|dog:0.05] [<lora:somelora:0.5:0.6>::0.5]
[in a park:in space:0.4]
```
The `example.json` contains a simple workflow to play around with.

## Schedulable LoRAs
The `LoRAScheduler` node patches a model such that when sampling, it'll switch LoRAs between steps. You can apply the LoRA's effect separately to CLIP conditioning and the unet (model)

For me this seems to be quite slow without the --highvram switch because ComfyUI will shuffle things between the CPU and GPU. YMMV. When things stay on the GPU, it's quite fast.

## AITemplate support
LoRA scheduling supports AITemplate. 

Due to sampler patching, your AITemplate nodes must be cloned to a directory called `AIT` under `custom_nodes` or the hijack won't find it.


# TODO & BUGS

The basics seem to work; without prompt editing, the loaders can reproduce the output from using `LoraLoader`

More advanced workflows might explode horribly.

- There are still some slight differences in LoRA output for some reason.
- Alternating does not work with LoRAs
- If execution is interrupted and LoRA scheduling is used, your models might be left in an undefined state until you restart ComfyUI
- Needs better syntax. A1111 is familiar, but not very good
- convenient prompt editing for multiple sampling passes (HR fix etc)
