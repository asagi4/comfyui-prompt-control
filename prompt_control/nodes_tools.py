import logging

from comfy_api.latest import io

from .parser import expand_macros, parse_prompt_schedules

log = logging.getLogger("comfyui-prompt-control")


class PCSetLogLevel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PCSetLogLevel",
            display_name="PC: Configure Logging (for debug)",
            category="promptcontrol/tools",
            description="A debug node to configure Prompt Control logging level. Pass a CLIP through it before you run any PC nodes",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input("level", options=["INFO", "DEBUG", "WARNING", "ERROR"], default="INFO", optional=True),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip, level="INFO") -> io.NodeOutput:
        log.setLevel(getattr(logging, level))
        log.info("Set logging level to %s", level)
        return io.NodeOutput(clip)


class PCAddMaskToCLIP(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PCAddMaskToCLIP",
            display_name="PC: Attach Mask",
            category="promptcontrol/tools",
            description="Attaches a mask to a CLIP object so that they can be referred to in a prompt using IMASK(). Using this node multiple times adds more masks rather than replacing existing ones.",
            inputs=[
                io.Clip.Input("clip"),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip, mask=None) -> io.NodeOutput:
        return PCAddMaskToCLIPMany.execute(clip, mask1=mask)


class PCAddMaskToCLIPMany(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PCAddMaskToCLIPMany",
            display_name="PC: Attach Mask (multi)",
            category="promptcontrol/tools",
            description="Multi-input version of PCAddMaskToCLIP, for convenience",
            inputs=[
                io.Clip.Input("clip"),
                io.Mask.Input("mask1", optional=True),
                io.Mask.Input("mask2", optional=True),
                io.Mask.Input("mask3", optional=True),
                io.Mask.Input("mask4", optional=True),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip, mask1=None, mask2=None, mask3=None, mask4=None) -> io.NodeOutput:
        clip = clip.clone()
        current_masks = clip.patcher.model_options.get("x-promptcontrol.masks", [])
        current_masks.extend(m for m in (mask1, mask2, mask3, mask4) if m is not None)
        clip.patcher.model_options["x-promptcontrol.masks"] = current_masks
        return io.NodeOutput(clip)


class PCSetPCTextEncodeSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PCSetPCTextEncodeSettings",
            display_name="PC: Configure PCTextEncode",
            category="promptcontrol/tools",
            description="Configures default values for PCTextEncode",
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input("mask_width", default=512, min=64, max=4096 * 4, optional=True),
                io.Int.Input("mask_height", default=512, min=64, max=4096 * 4, optional=True),
                io.Int.Input("sdxl_width", default=1024, min=0, max=4096 * 4, optional=True),
                io.Int.Input("sdxl_height", default=1024, min=0, max=4096 * 4, optional=True),
                io.Int.Input("sdxl_target_w", default=1024, min=0, max=4096 * 4, optional=True),
                io.Int.Input("sdxl_target_h", default=1024, min=0, max=4096 * 4, optional=True),
                io.Int.Input("sdxl_crop_w", default=0, min=0, max=4096 * 4, optional=True),
                io.Int.Input("sdxl_crop_h", default=0, min=0, max=4096 * 4, optional=True),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(
        cls,
        clip,
        mask_width=512,
        mask_height=512,
        sdxl_width=1024,
        sdxl_height=1024,
        sdxl_target_w=1024,
        sdxl_target_h=1024,
        sdxl_crop_w=0,
        sdxl_crop_h=0,
    ) -> io.NodeOutput:
        settings = {
            "mask_width": mask_width,
            "mask_height": mask_height,
            "sdxl_width": sdxl_width,
            "sdxl_height": sdxl_height,
            "sdxl_twidth": sdxl_target_w,
            "sdxl_theight": sdxl_target_h,
            "sdxl_cwidth": sdxl_crop_w,
            "sdxl_cheight": sdxl_crop_h,
        }
        clip = clip.clone()
        clip.patcher.model_options["x-promptcontrol.settings"] = settings
        return io.NodeOutput(clip)


class PCExtractScheduledPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PCExtractScheduledPrompt",
            display_name="PC: Extract Scheduled Prompt",
            category="promptcontrol/tools",
            description="Parses the input prompt and returns the prompt scheduled at the specified point",
            inputs=[
                io.String.Input("text", multiline=True),
                io.Float.Input("at", min=0.0, max=1.0, default=1.0, step=0.01),
                io.String.Input("tags", default="", optional=True),
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, text, at, tags="") -> io.NodeOutput:
        schedule = parse_prompt_schedules(text, filters=tags)
        _, entry = schedule.at_step(at, total_steps=1)
        prompt_text = entry.get("prompt", "")
        return io.NodeOutput(prompt_text)


class PCMacroExpand(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PCMacroExpand",
            display_name="PC: Expand Macros",
            category="promptcontrol/tools",
            description="Expands DEF macros in a string and returns the result",
            inputs=[
                io.String.Input("text", multiline=True),
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        return io.NodeOutput(expand_macros(text))


NODES = [
    PCSetPCTextEncodeSettings,
    PCAddMaskToCLIP,
    PCAddMaskToCLIPMany,
    PCSetLogLevel,
    PCExtractScheduledPrompt,
    PCMacroExpand,
]
