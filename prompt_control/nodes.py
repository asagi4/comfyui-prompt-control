from . import utils as utils
import comfy.sd
from .utils import Timer
from .parser import parse_prompt_schedules

log = utils.getlogger()

def apply_loras_to_model(model, orig_model, clip, lora_specs, loaded_loras):
    key_map = comfy.sd.model_lora_keys_unet(model.model)
    key_map = comfy.sd.model_lora_keys_clip(clip.cond_stage_model, key_map)

    with Timer("Unpatch model"):
        model.unpatch_model()
        model = orig_model.clone()

    for name, weights in lora_specs:
        if name not in loaded_loras:
            continue
        loaded = comfy.sd.load_lora(loaded_loras[name], key_map)
        model = model.clone()
        model.add_patches(loaded, weights[0])
        log.info("Loaded LoRA %s:%s", name, weights[0])

    with Timer("Repatch model"):
        model.patch_model()
    return model

class LoRAScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "clip": ("CLIP",),
                 "text": ("STRING", {"multiline": True}),
                }
               }
    RETURN_TYPES = ('MODEL',)
    CATEGORY = 'promptcontrol'
    FUNCTION = 'apply'

    def apply(self, model, clip, text):
        orig_model = model.clone()
        schedules = parse_prompt_schedules(text)
        log.debug("LoRAScheduler: %s", schedules)
        loaded_loras = {}
        loaded_loras = utils.load_loras_from_schedule(schedules, loaded_loras)

        def sampler_cb(orig_sampler, *args, **kwargs):
            state = {}
            steps = args[2]
            start_step = kwargs['start_step'] or 0
            # The model patcher may change if LoRAs are applied
            state['model'] = args[0]
            state['applied_loras'] = []

            orig_cb = kwargs['callback']
            def apply_lora_for_step(step):
                # zero-indexed steps, 0 = first step, but schedules are 1-indexed
                sched = utils.schedule_for_step(steps, step+1, schedules)
                lora_spec = sorted(sched[1]['loras'])
                    
                if state['applied_loras'] != lora_spec:
                    log.debug("At step %s, applying lora_spec %s", step, lora_spec)
                    state['model'] = apply_loras_to_model(state['model'], orig_model, clip, lora_spec, loaded_loras)
                    state['applied_loras'] = lora_spec

            def step_callback(*args, **kwargs):
                current_step = args[0] + start_step
                # Callbacks are called *after* the step so apply for next step
                apply_lora_for_step(current_step+1)
                if orig_cb:
                    return orig_cb(*args, **kwargs)

            kwargs['callback'] = step_callback

            apply_lora_for_step(start_step)
            s = orig_sampler(*args, **kwargs)

            if state['applied_loras']:
                log.info("Sampling done with leftover LoRAs, unpatching")
                # state may have been modified
                state['model'].unpatch_model()

            return s

        setattr(orig_model, 'prompt_control_callback', sampler_cb)
        return (orig_model,)

class EditableCLIPEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"clip": ("CLIP",),
                 "model": ("MODEL",),
                 "text": ("STRING", {"multiline": True}),
                }
               }
    RETURN_TYPES = ('CONDITIONING',)
    CATEGORY = 'promptcontrol'
    FUNCTION = 'parse'

    def __init__(self):
        self.loaded_loras = {}
        self.current_loras = []
        self.orig_clip = None

    def load_clip_lora(self, clip, model, loraspec):
        if not loraspec:
            return clip
        key_map = comfy.sd.model_lora_keys_unet(model.model)
        key_map = comfy.sd.model_lora_keys_clip(clip.cond_stage_model, key_map)
        if self.current_loras != loraspec:
            for l in loraspec:
                name, w = l
                w = w + [w[0]]
                if name not in self.loaded_loras:
                    log.warn("%s not loaded, skipping", name)
                    continue
                loaded = comfy.sd.load_lora(self.loaded_loras[name], key_map)
                clip.add_patches(loaded, w[1])
                log.info("CLIP LoRA loaded: %s:%s", name, w[1])
        return clip

    def parse(self, clip, model, text):
        parsed = parse_prompt_schedules(text)
        log.debug("EditableCLIPEncode schedules: %s", parsed)
        self.current_loras = []
        self.loaded_loras = utils.load_loras_from_schedule(parsed, self.loaded_loras)
        self.orig_clip = clip
        start_pct = 0.0
        conds = []
        for end_pct, c in parsed:
            if c["loras"] != self.current_loras:
                clip = self.load_clip_lora(self.orig_clip.clone(), model, c["loras"])
                self.current_loras = c["loras"]
            tokens = clip.tokenize(c["prompt"])
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conds.append([cond, {"pooled_output": pooled, "start_percent": 1.0 - start_pct, "end_percent": 1.0 - end_pct}])
            start_pct = end_pct
        return (conds,)

NODE_CLASS_MAPPINGS = {
    "EditableCLIPEncode": EditableCLIPEncode,
    "LoRAScheduler": LoRAScheduler,
}
