# Adapted from ComfyUI-ppm into hook form


import comfy.model_management
import comfy.patcher_extension
from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
from comfy.model_base import Anima
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io

from .anima_couple import (
    anima_forward_wrapper,
    anima_sample_wrapper,
    cosmos_attention_forward_couple,
)


class CoupleForward:
    def __init__(self, fn, block):
        self.fn = fn
        self.block = block

    def __call__(self, *args, **kwargs):
        transformer_options = kwargs["transformer_options"]
        pc = transformer_options.get("pc_couple")
        if pc:
            self.block.to("cuda")
        return cosmos_attention_forward_couple(self.fn, *args, **kwargs)


class PCAnimaAttnCouplePatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PCAnimaAttnCouplePatch",
            display_name="PC: Anima attention Couple Model Patch",
            category="promptcontrol/experimental",
            inputs=[
                io.Model.Input("model"),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model: ModelPatcher) -> io.NodeOutput:
        model_type = type(model.model)
        m = model

        if issubclass(model_type, Anima):
            m = model.clone()
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                cls.__name__,
                anima_forward_wrapper,
            )
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
                cls.__name__,
                anima_sample_wrapper,
            )

        return io.NodeOutput(m)


NODES = [PCAnimaAttnCouplePatch]
