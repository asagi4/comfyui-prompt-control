import logging

from comfy_api.latest import io

from .prompts import encode_prompt

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
    def execute(cls, clip, text, start=0.0, end=1.0) -> io.NodeOutput:  # ty: ignore[invalid-method-override]
        log.debug("PCTextEncode: Encoding '%s'", text)
        defaults = clip.patcher.model_options.get("x-promptcontrol.defaults", {})
        masks = clip.patcher.model_options.get("x-promptcontrol.masks", None)
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
    def execute(cls, clip, text) -> io.NodeOutput:  # ty: ignore[invalid-method-override]
        # Use the WithRange node for the range 0.0, 1.0
        return PCTextEncodeWithRange.execute(clip, text, 0.0, 1.0)


NODES = [
    PCTextEncodeWithRange,
    PCTextEncode,
]
