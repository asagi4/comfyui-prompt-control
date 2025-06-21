# PC: Configure PCTextEncode

Configures a CLIP object with new default values used by `PCTextEncode`. Apply it before everything else.

This is needed if you want to do scheduling with steps instead of denoising percentages, but otherwise it's completely optional.

Note that steps are simply syntactic sugar for percentages and may not correspond to actual steps depending on the scheduler used.
