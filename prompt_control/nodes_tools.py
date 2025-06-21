import logging
from .parser import parse_prompt_schedules, expand_macros
from .nodes_lazy import NODE_CLASS_MAPPINGS as LAZY_NODES
from .utils import expand_graph
import json
import folder_paths
from pathlib import Path

log = logging.getLogger("comfyui-prompt-control")


class PCSaveExpandedWorkflow:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*", {}),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return True

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    CATEGORY = "promptcontrol/tools"
    DESCRIPTION = "Expands lazy prompt control nodes in the prompt and saves the expanded prompt into a JSON file"

    FUNCTION = "apply"

    def apply(self, any, prompt):
        full_output_folder, filename, counter, subfolder, prefix = folder_paths.get_save_image_path(
            "pc_workflow_debug", self.output_dir
        )
        expanded = expand_graph(LAZY_NODES, prompt)
        file = f"{filename}_{counter:05}_.json"
        full_path = Path(full_output_folder) / file
        with open(full_path, "w") as f:
            log.info(f"Saving workflow to {full_path}")
            json.dump(expanded, f)

        return ()


class PCSetLogLevel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
            },
            "optional": {
                "level": (["INFO", "DEBUG", "WARNING", "ERROR"], {"default": "INFO"}),
            },
        }

    def apply(self, clip, level="INFO"):
        log.setLevel(getattr(logging, level))
        log.info("Set logging level to %s", level)
        return (clip,)

    RETURN_TYPES = ("CLIP",)
    CATEGORY = "promptcontrol/tools"
    DESCRIPTION = (
        "A debug node to configure Prompt Control logging level. Pass a CLIP through it before you run any PC nodes"
    )

    FUNCTION = "apply"


class PCAddMaskToCLIP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP",)},
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("CLIP",)
    CATEGORY = "promptcontrol/tools"
    FUNCTION = "apply"
    DESCRIPTION = "Attaches a mask to a CLIP object so that they can be referred to in a prompt using IMASK(). Using this node multiple times adds more masks rather than replacing existing ones."

    def apply(self, clip, mask=None):
        return PCAddMaskToCLIPMany().apply(clip, mask1=mask)


class PCAddMaskToCLIPMany:
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
    CATEGORY = "promptcontrol/tools"
    FUNCTION = "apply"
    DESCRIPTION = "Multi-input version of PCAddMaskToCLIP, for convenience"

    def apply(self, clip, mask1=None, mask2=None, mask3=None, mask4=None):
        clip = clip.clone()
        current_masks = clip.patcher.model_options.get("x-promptcontrol.masks", [])
        current_masks.extend(m for m in (mask1, mask2, mask3, mask4) if m is not None)
        clip.patcher.model_options["x-promptcontrol.masks"] = current_masks
        return (clip,)


class PCSetPCTextEncodeSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP",)},
            "optional": {
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
    CATEGORY = "promptcontrol/tools"
    FUNCTION = "apply"
    DESCRIPTION = "Configures default values for PCTextEncode"

    def apply(
        self,
        clip,
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
        return (clip,)


class PCExtractScheduledPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "at": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.01}),
            },
            "optional": {"tags": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "promptcontrol/tools"
    FUNCTION = "apply"
    DESCRIPTION = "Parses the input prompt and returns the prompt scheduled at the specified point"

    def apply(self, text, at, tags=""):
        schedule = parse_prompt_schedules(text, filters=tags)
        _, entry = schedule.at_step(at, total_steps=1)
        prompt_text = entry.get("prompt", "")
        return (prompt_text,)


class PCMacroExpand:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "promptcontrol/tools"
    FUNCTION = "apply"
    DESCRIPTION = "Expands DEF macros in a string and returns the result"

    def apply(self, text):
        return (expand_macros(text),)


NODE_CLASS_MAPPINGS = {
    "PCSetPCTextEncodeSettings": PCSetPCTextEncodeSettings,
    "PCAddMaskToCLIP": PCAddMaskToCLIP,
    "PCAddMaskToCLIPMany": PCAddMaskToCLIPMany,
    "PCSetLogLevel": PCSetLogLevel,
    "PCExtractScheduledPrompt": PCExtractScheduledPrompt,
    "PCSaveExpandedWorkflow": PCSaveExpandedWorkflow,
    "PCMacroExpand": PCMacroExpand,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PCSetPCTextEncodeSettings": "PC: Configure PCTextEncode",
    "PCAddMaskToCLIP": "PC: Attach Mask",
    "PCAddMaskToCLIPMany": "PC: Attach Mask (multi)",
    "PCSetLogLevel": "PC: Configure Logging (for debug)",
    "PCExtractScheduledPrompt": "PC: Extract Scheduled Prompt",
    "PCSaveExpandedWorkflow": "PC: Save Expanded Workflow (for debug)",
    "PCMacroExpand": "PC: Expand Macros",
}
