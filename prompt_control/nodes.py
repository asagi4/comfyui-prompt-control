from . import utils as utils
import comfy.sd
from .utils import untuple
from .parser import parse_prompt_schedules
from .hijack import do_hijack, get_aitemplate_module

log = utils.getlogger()


# AITemplate support
def set_callback(model, cb):
    model = untuple(model)
    setattr(model, "prompt_control_callback", cb)


def get_lora_keymap(model, clip):
    model = untuple(model)
    key_map = comfy.sd.model_lora_keys_unet(model.model)
    return comfy.sd.model_lora_keys_clip(clip.cond_stage_model, key_map)


def unpatch_model(model):
    untuple(model).unpatch_model()


def clone_model(model):
    if isinstance(model, tuple):
        return (model[0].clone(), model[1])
    else:
        return model.clone()


def add_patches(model, patches, weight):
    untuple(model).add_patches(patches, weight)


def patch_model(model):
    if isinstance(model, tuple):
        m = model[0]
        m.patch_model()
        mod = get_aitemplate_module()
        l = mod.AITemplate.loader
        if hasattr(l, "pc_applied_module"):
            log.info("Applying AITemplate unet")
            l.apply_unet(
                aitemplate_module=l.pc_applied_module,
                unet=l.compvis_unet(m.model.state_dict()),
                in_channels=m.model.diffusion_model.in_channels,
                conv_in_key="conv_in_weight",
            )
    else:
        model.patch_model()


def load_lora(model, lora, weight, key_map):
    loaded = comfy.sd.load_lora(lora, key_map)
    model = clone_model(model)
    add_patches(model, loaded, weight)
    return model


def apply_loras_to_model(model, orig_model, clip, lora_specs, loaded_loras, patch=True):
    keymap = get_lora_keymap(model, clip)
    if patch:
        unpatch_model(model)
        model = clone_model(model)

    for name, weights in lora_specs:
        if name not in loaded_loras:
            continue
        model = load_lora(model, loaded_loras[name], weights[0], keymap)
        log.info("Loaded LoRA %s:%s", name, weights[0])

    if patch:
        patch_model(model)

    return model


class LoRAScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, model, clip, text):
        do_hijack()
        orig_model = clone_model(model)
        schedules = parse_prompt_schedules(text)
        log.debug("LoRAScheduler: %s", schedules)
        loaded_loras = {}
        loaded_loras = utils.load_loras_from_schedule(schedules, loaded_loras)

        def sampler_cb(orig_sampler, *args, **kwargs):
            state = {}
            steps = args[2]
            start_step = kwargs["start_step"] or 0
            # The model patcher may change if LoRAs are applied
            state["model"] = args[0]
            state["applied_loras"] = []

            orig_cb = kwargs["callback"]

            def apply_lora_for_step(step, patch=False):
                # zero-indexed steps, 0 = first step, but schedules are 1-indexed
                sched = utils.schedule_for_step(steps, step + 1, schedules)
                lora_spec = sorted(sched[1]["loras"])

                if state["applied_loras"] != lora_spec:
                    log.debug("At step %s, applying lora_spec %s", step, lora_spec)
                    state["model"] = apply_loras_to_model(
                        state["model"], orig_model, clip, lora_spec, loaded_loras, patch
                    )
                    state["applied_loras"] = lora_spec

            def step_callback(*args, **kwargs):
                current_step = args[0] + start_step
                # Callbacks are called *after* the step so apply for next step
                apply_lora_for_step(current_step + 1)
                if orig_cb:
                    return orig_cb(*args, **kwargs)

            kwargs["callback"] = step_callback

            # First step of sampler applies patch
            apply_lora_for_step(start_step, patch=False)
            args = list(args)
            args[0] = state["model"]
            s = orig_sampler(*args, **kwargs)

            if state["applied_loras"]:
                log.info("Sampling done with leftover LoRAs, unpatching")
                # state may have been modified
                unpatch_model(state["model"])

            return s

        set_callback(orig_model, sampler_cb)

        return (orig_model,)


class EditableCLIPEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "model": ("MODEL",),
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol"
    FUNCTION = "parse"

    def __init__(self):
        self.loaded_loras = {}
        self.current_loras = []
        self.orig_clip = None

    def load_clip_lora(self, clip, model, loraspec):
        if not loraspec:
            return clip
        key_map = get_lora_keymap(model, clip)
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
            conds.append(
                [
                    cond,
                    {
                        "pooled_output": pooled,
                        "start_percent": 1.0 - start_pct,
                        "end_percent": 1.0 - end_pct,
                        "prompt": c["prompt"],
                    },
                ]
            )
            start_pct = end_pct
        return (conds,)


class ConditioningCutoff:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conds": ("CONDITIONING",),
                "cutoff": ("FLOAT", {"min": 0.00, "max": 1.00, "default": 0.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, conds, cutoff):
        res = []
        new_start = 1.0
        for c in conds:
            start = c[1].get("start_percent", 1.0)
            end = c[1].get("end_percent", 0.0)
            if 1.0 - end < cutoff:
                log.debug("Chose to remove prompt '%s'", c[1].get("prompt", "N/A"))
                continue
            c = [c[0].clone(), c[1].copy()]
            c[1]["start_percent"] = new_start
            c[1]["end_percent"] = end
            new_start = end
            res.append(c)

        log.debug("Conds after filter: %s", [(c[1]["prompt"], c[1]["start_percent"], c[1]["end_percent"]) for c in res])
        return (res,)


NODE_CLASS_MAPPINGS = {
    "EditableCLIPEncode": EditableCLIPEncode,
    "LoRAScheduler": LoRAScheduler,
    "ConditioningCutoff": ConditioningCutoff,
}
