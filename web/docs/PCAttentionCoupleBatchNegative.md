# PC: Attention Couple (batch negative)

This node applies an optimization that re-enables negative cond batching when Attention Couple is in use.

It improves performance when negative prompts are not scheduled, but slightly affects outputs and is not required for Attention Couple to work.

Simply add it to your workflow and pass in your positive and negative prompts. It is always safe to use, as it will not do anything when it detects that the optimization can't be applied (eg. when negative prompts contain schedules)
