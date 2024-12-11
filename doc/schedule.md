# Nodes operating on schedules

These nodes operate on special "schedule" objects. They don't benefit from laziness like `PCLazyTextEncode`, but allow attaching custom masks.

## PCLoraHooksFromSchedule

Creates a ComfyUI `HOOKS` object from a prompt schedule. Can be attached to a CLIP model to perform encoding and LoRA switching.

## PCEncodeSchedule

Encodes all prompts in a schedule. Pass in a `CLIP` object with hooks attached for LoRA scheduling, then use the resulting `CONDITIONING` normally

## PCPromptToSchedule
Parses a schedule from a text prompt. A schedule is essentially an array of `(valid_until, prompt)` pairs that the other nodes can use.

## PCFilterSchedule
Filters a schedule according to its parameters, removing any *changes* that do not occur within `[start, end)`.

The node also does tag filtering if any tags are specified.

Always returns at least the last prompt in the schedule if everything would otherwise be filtered.

`start=0, end=0` returns the prompt at the start and `start=1.0, end=1.0` returns the prompt at the end.

## PCSettings
Returns an object representing **default values** for the `SDXL` function and allows configuring `MASK_SIZE` outside the prompt. You need to apply them to a schedule with `PCApplySettings`. Note that for the SDXL settings to apply, you still need to have `SDXL()` in the prompt.

The "steps" parameter currently does nothing; it's for future features.

## PCApplySettings
Applies the given default values from `PCSettings` to a schedule

## PCPromptFromSchedule

Extracts a text prompt from a schedule; also logs it to the console.
LoRAs are *not* included in the text prompt, though they are logged.

## PCScheduleAddMasks

Add masks to a schedule object, for use with IMASK
