# Adapted from ComfyUI-ppm into hook form


import comfy.model_management
import comfy.patcher_extension
from comfy.model_base import Anima
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io

from .anima_couple import (
    anima_forward_wrapper,
    anima_sample_wrapper,
)


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
