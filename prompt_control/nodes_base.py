import logging
from .prompts import encode_prompt

log = logging.getLogger("comfyui-prompt-control")


class PCTextEncodeWithRange:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP",), "text": ("STRING", {"multiline": True})},
            "optional": {
                "start": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.01}),
                "end": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol/tools"
    FUNCTION = "apply"
    DESCRIPTION = "Like PCTextEncode, but if you know the range you need for a prompt, can be slightly more efficient when you have LoRAs scheduled on a CLIP model"

    def apply(self, clip, text, start=0.0, end=1.0):
        log.debug("PCTextEncode: Encoding '%s'", text)
        defaults = clip.patcher.model_options.get("x-promptcontrol.defaults", {})
        masks = clip.patcher.model_options.get("x-promptcontrol.masks", None)
        return (encode_prompt(clip, text, start, end, defaults, masks),)


class PCTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP",), "text": ("STRING", {"multiline": True})},
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"
    DESCRIPTION = "Encodes a prompt with extra goodies from Prompt Control. This node does *not* support scheduling"

    def apply(self, clip, text):
        return PCTextEncodeWithRange.apply(self, clip, text, 0.0, 1.0)


NODE_CLASS_MAPPINGS = {"PCTextEncode": PCTextEncode, "PCTextEncodeWithRange": PCTextEncodeWithRange}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PCTextEncode": "PC: Text Encode (no scheduling)",
    "PCTextEncodeWithRange": "PC: Text Encode with Range (no scheduling)",
}
