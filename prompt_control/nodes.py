from . import utils as utils
import comfy.sd
from .utils import Timer
from .parser import parse_prompt_schedules

log = utils.getlogger()

def apply_loras_to_model(s, lora_specs, loaded_loras):
    model = s['model']
    clip = s['clip']
    key_map = comfy.sd.model_lora_keys_unet(model.model)
    key_map = comfy.sd.model_lora_keys_clip(clip.cond_stage_model, key_map)

    with Timer("Unpatch model"):
        model.unpatch_model()
        model = s['orig_model'].clone()

    for name, weights in lora_specs:
        if name not in loaded_loras:
            continue
        loaded = comfy.sd.load_lora(loaded_loras[name], key_map)
        model = model.clone()
        model.add_patches(loaded, weights[0])
        log.info("Loaded LoRA %s:%s", name, weights[0])

    with Timer("Repatch model"):
        model.patch_model()
    s['applied_loras'] = sorted(lora_specs)
    s['model'] = model

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

    def __init__(self):
        self.orig_model = None
        self.orig_text = None

    def apply(self, model, clip, text):
        model = model.clone()

        def sampler_cb(*args, **kwargs):
            state = {}
            # Store the original in case we need to reset
            # We need this for LoRA keys
            state['clip'] = clip
            state['orig_model'] = args[0]
            state['model'] = args[0]
            state['steps'] = args[2] # steps
            state['current_step'] = kwargs['start_step'] or 1
            state['last_step'] = min(state['steps'], kwargs['last_step'] or state['steps'])
            if not 'applied_loras' in state:
                state['applied_loras'] = []

            setattr(model, 'prompt_control_state', state)
            
            return (args, kwargs)

        setattr(model, 'prompt_control_callback', sampler_cb)


        schedules = parse_prompt_schedules(text)
        loaded_loras = {}
        loaded_loras = utils.load_loras_from_schedule(schedules, loaded_loras)

        orig_model_fn = model.model_options.get('model_function_wrapper')
        def unet_wrapper(model_fn, params):
            s = model.prompt_control_state
            sched = utils.schedule_for_step(s['steps'], s['current_step'], schedules)
            lora_spec = sorted(sched[1]['loras'])
                
            if s['applied_loras'] != lora_spec:
                apply_loras_to_model(s, lora_spec, loaded_loras)
                s['applied_loras'] = lora_spec
            s['current_step'] += 1

            if not orig_model_fn:
                r = model_fn(params['input'], params['timestep'], **params['c'])
            else:
                r = orig_model_fn(model_fn, params)

            if s['current_step'] > s['last_step']:
                if s["applied_loras"]:
                    log.info("sampling done, unpatching model")
                    s["model"].unpatch_model()
            return r

        model.set_model_unet_function_wrapper(unet_wrapper)

        return (model,)

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
            clip = clip.clone()
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
        self.current_loras = []
        self.loaded_loras = utils.load_loras_from_schedule(parsed, self.loaded_loras)
        self.orig_clip = clip
        start_pct = 0.0
        conds = []
        for end_pct, c in parsed:
            if c["loras"] != self.current_loras:
                clip = self.load_clip_lora(clip, model, c["loras"])
                self.current_loras = c["loras"]
            conds.append([clip.encode(c["prompt"]), {"start_percent": 1.0 - start_pct, "end_percent": 1.0 - end_pct}])
            start_pct = end_pct
        return (conds,)

NODE_CLASS_MAPPINGS = {
    "EditableCLIPEncode": EditableCLIPEncode,
    "LoRAScheduler": LoRAScheduler,
}
