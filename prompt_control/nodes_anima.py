# Adapted from ComfyUI-ppm into hook form

from functools import partial

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
            anima_model = model.get_model_object("diffusion_model")
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

            for block_name, _ in (
                (n, b) for n, b in anima_model.named_modules() if "cross_attn" in n and isinstance(b, CosmosAttention)
            ):
                attn_forward_prev = m.get_model_object(f"diffusion_model.{block_name}.forward")
                m.add_object_patch(
                    f"diffusion_model.{block_name}.forward", partial(cosmos_attention_forward_couple, attn_forward_prev)
                )

        return io.NodeOutput(m)


NODES = [PCAnimaAttnCouplePatch]
