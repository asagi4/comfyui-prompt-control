import logging

log = logging.getLogger("comfyui-prompt-control")


class PCAddMasksToCLIP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP",)},
            "optional": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "mask3": ("MASK",),
                "mask4": ("MASK",),
            },
        }

    RETURN_TYPES = ("CLIP",)
    CATEGORY = "promptcontrol/v2"
    FUNCTION = "apply"

    def apply(self, clip, mask1=None, mask2=None, mask3=None, mask4=None):
        clip = clip.clone()
        current_masks = clip.patcher.model_options.get("x-pc.masks", [])
        current_masks.extend(m for m in (mask1, mask2, mask3, mask4) if m is not None)
        clip.patcher.model_options["x-promptcontrol.masks"] = current_masks
        return (clip,)


class PCSetPCTextEncodeSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP",)},
            "optional": {
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "mask_width": ("INT", {"default": 512, "min": 64, "max": 4096 * 4}),
                "mask_height": ("INT", {"default": 512, "min": 64, "max": 4096 * 4}),
                "sdxl_width": ("INT", {"default": 1024, "min": 0, "max": 4096 * 4}),
                "sdxl_height": ("INT", {"default": 1024, "min": 0, "max": 4096 * 4}),
                "sdxl_target_w": ("INT", {"default": 1024, "min": 0, "max": 4096 * 4}),
                "sdxl_target_h": ("INT", {"default": 1024, "min": 0, "max": 4096 * 4}),
                "sdxl_crop_w": ("INT", {"default": 0, "min": 0, "max": 4096 * 4}),
                "sdxl_crop_h": ("INT", {"default": 0, "min": 0, "max": 4096 * 4}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    CATEGORY = "promptcontrol/v2"
    FUNCTION = "apply"

    def apply(
        self,
        clip,
        steps=0,
        mask_width=512,
        mask_height=512,
        sdxl_width=1024,
        sdxl_height=1024,
        sdxl_target_w=1024,
        sdxl_target_h=1024,
        sdxl_crop_w=0,
        sdxl_crop_h=0,
    ):
        settings = {
            "steps": steps,
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
        clip.patcher.model_options["x-promptcontrol.default_settings"] = settings
        return (clip,)


NODE_CLASS_MAPPINGS = {
    "PCSetPCTextEncodeSettings": PCSetPCTextEncodeSettings,
    "PCAddMasksToCLIP": PCAddMasksToCLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PCSetTextEncodeSettings": "PC Set PCTextEncode Defaults",
    "PCAddMasksToCLIP": "PC Attach Custom Masks (for IMASK)",
}
