# Legacy node documentation

You really shouldn't be using these anymore

## Old nodes
The `ScheduleToModel` node patches a model so that when sampling, it'll switch LoRAs between steps. You can apply the LoRA's effect separately to CLIP conditioning and the unet (model).

Swapping LoRAs often can be quite slow without the `--highvram` switch because ComfyUI will shuffle things between the CPU and GPU. When things stay on the GPU, it's quite fast.

If you run out of VRAM during a LoRA swap, the node will attempt to save VRAM by enabling CPU offloading for future generations even in highvram mode. This persists until ComfyUI is restarted.

You can also set the `PC_RETRY_ON_OOM` environment variable to any non-empty value to automatically retry sampling once if VRAM runs out.

## ScheduleToCond (deprecated)
Produces a combined conditioning for the appropriate timesteps. From a schedule. Also applies LoRAs to the CLIP model according to the schedule.

## ScheduleToModel (deprecated)
Produces a model that'll cause the sampler to reapply LoRAs at specific steps according to the schedule.

This depends on a callback handled by a monkeypatch of the ComfyUI sampler function, so it might not work with custom samplers, but it shouldn't interfere with them either.

## PCSplitSampling (deprecated)
Causes sampling to be split into multiple sampler calls instead of relying on timesteps for scheduling. This makes the schedules more accurate, but seems to cause weird behaviour with SDE samplers. (Upstream bug?)


## PromptControlSimple (deprecated)
This node exists purely for convenience. It's a combination of `PromptToSchedule`, `ScheduleToCond`, `ScheduleToModel` and `FilterSchedule` such that it provides as output a model, positive conds and negative conds, both with and without any specified filters applied.

This makes it handy for quick one- or two-pass workflows.

## Older nodes

- `EditableCLIPEncode`: A combination of `PromptToSchedule` and `ScheduleToCond`
- `LoRAScheduler`: A combination of `PromptToSchedule`, `FilterSchedule` and `ScheduleToModel`

# Known issues

- If you use LoRA scheduling in a workflow with `LoRALoader` nodes, you might get inconsistent results. For now, just avoid mixing `ScheduleToModel` or `LoRAScheduler` with `LoRALoader`. See https://github.com/asagi4/comfyui-prompt-control/issues/36
- Workflows using `SamplerCustom` will calculate LoRA schedules based on the number of sigmas given to the sampler instead of the number of steps, since that information isn't available.
- `CUT` does not work with `STYLE:perp`
- `PCSplitSampling` overrides ComfyUI's `BrownianTreeNoiseSampler` noise sampling behaviour so that each split segment doesn't add crazy amounts of noise to the result with some samplers.
- Split sampling may have weird behaviour if your step percentages go below 1 step.
- Interpolation is probably buggy and will likely change behaviour whenever code gets refactored.
- If execution is interrupted and LoRA scheduling is used, your models might be left in an undefined state until you restart ComfyUI
