import logging
from .prompts import encode_prompt

log = logging.getLogger("comfyui-prompt-control")


class PCTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP",), "text": ("STRING", {"multiline": True})},
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, clip, text):
        defaults = clip.patcher.model_options.get("x-promptcontrol.defaults", {})
        masks = clip.patcher.model_options.get("x-promptcontrol.masks", None)
        return (encode_prompt(clip, text, 0, 1.0, defaults, masks),)


NODE_CLASS_MAPPINGS = {
    "PCTextEncode": PCTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PCTextEncode": "PC Text Encode (no scheduling)",
}
