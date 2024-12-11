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

    def apply(self, clip, text, defaults=None):
        return (encode_prompt(clip, text, 0, 1.0, defaults or {}, None),)


NODE_CLASS_MAPPINGS = {
    "PCTextEncode": PCTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PCTextEncode": "PC Text Encode (no scheduling)",
}
