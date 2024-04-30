from .node_clip import control_to_clip_common
from .node_lora import schedule_lora_common
from .parser import parse_prompt_schedules


class PromptControlSimple:
    cached_model = None
    cached_clone = None
    lora_cache = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "positive": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True}),
            },
            "optional": {
                "tags": ("STRING", {"default": ""}),
                "start": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.1, "default": 0.0}),
                "end": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.1, "default": 1.0}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative", "model_filtered", "pos_filtered", "neg_filtered")
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, model, clip, positive, negative, tags="", start=0.0, end=1.0):
        if model != self.cached_model:
            self.cached_model = model
            self.cached_clone = model.clone()
            self.lora_cache = {}
        cond_cache = {}
        pos_sched = parse_prompt_schedules(positive)
        pos_cond = pos_filtered = control_to_clip_common(clip, pos_sched, self.lora_cache, cond_cache)

        neg_sched = parse_prompt_schedules(negative)
        neg_cond = neg_filtered = control_to_clip_common(clip, neg_sched, self.lora_cache, cond_cache)

        new_model = model_filtered = schedule_lora_common(self.cached_clone, pos_sched, self.lora_cache)

        if [tags.strip(), start, end] != ["", 0.0, 1.0]:
            pos_filtered = control_to_clip_common(
                clip, pos_sched.with_filters(tags, start, end), self.lora_cache, cond_cache
            )
            neg_filtered = control_to_clip_common(
                clip, neg_sched.with_filters(tags, start, end), self.lora_cache, cond_cache
            )
            model_filtered = schedule_lora_common(
                self.cached_clone, pos_sched.with_filters(tags, start, end), self.lora_cache
            )

        return (new_model, pos_cond, neg_cond, model_filtered, pos_filtered, neg_filtered)
