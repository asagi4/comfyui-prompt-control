import logging

from comfy_api.latest import io

from .macros import expand_segs
from .prompts import encode_prompt, hook_te

log = logging.getLogger("comfyui-prompt-control")


class PCTextEncodeWithRange(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PCTextEncodeWithRange",
            display_name="PC: Text Encode with Range (no scheduling)",
            category="promptcontrol/tools",
            description="Like PCTextEncode, but if you know the range you need for a prompt, can be slightly more efficient when you have LoRAs scheduled on a CLIP model.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("text", multiline=True),
                io.Float.Input("start", default=0.0, min=0.0, max=1.0, step=0.01, optional=True),
                io.Float.Input("end", default=1.0, min=0.0, max=1.0, step=0.01, optional=True),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, text, start=0.0, end=1.0) -> io.NodeOutput:
        log.debug("PCTextEncode: Encoding '%s'", text)
        defaults = clip.patcher.model_options.get("x-promptcontrol.defaults", {})
        masks = clip.patcher.model_options.get("x-promptcontrol.masks", None)
        text = expand_segs(text)
        out = encode_prompt(clip, text, start, end, defaults, masks)
        return io.NodeOutput(out)


class PCTextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PCTextEncode",
            display_name="PC: Text Encode (no scheduling)",
            category="promptcontrol",
            description="Encodes a prompt with extra goodies from Prompt Control. This node does *not* support scheduling.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("text", multiline=True),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, text) -> io.NodeOutput:
        # Use the WithRange node for the range 0.0, 1.0
        return PCTextEncodeWithRange.execute(clip, text, 0.0, 1.0)


class PCHookEncoderModsInternal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PCHookTextEncoderModsInternal",
            display_name="PC: Apply Text Encoder Mods",
            category="promptcontrol",
            description="Apply TE modifications (internal)",
            is_experimental=True,
            is_dev_only=True,
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("te_names"),
                io.String.Input("style"),
                io.String.Input("normalization"),
                io.Custom("PC_EXTRA_DATA").Input("extra", optional=True),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip, te_names, style, normalization, extra) -> io.NodeOutput:
        te_names = [x.strip() for x in te_names.split(",")]
        clip = hook_te(clip, te_names, style, normalization, extra)
        return io.NodeOutput(clip)


NODES = [PCTextEncodeWithRange, PCTextEncode, PCHookEncoderModsInternal]
