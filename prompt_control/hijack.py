from .utils import getlogger
import comfy.sample
import sys

log = getlogger()
orig_sampler = comfy.sample.sample

# AITemplate support
def have_aitemplate():
    global hijack_aitemplate
    return hijack_aitemplate

def get_callback(model):
    if isinstance(model, tuple):
        model = model[0]
    return getattr(model, 'prompt_control_callback', None)

def do_hijack():
    if hasattr(comfy.sample.sample, 'pc_hijack_done'):
        return
    def pc_sample(*args, **kwargs):
        print("Hijack called, AITemplate:")
        model = args[0]
        cb = get_callback(model)
        if cb:
            return cb(orig_sampler, *args, **kwargs)
        else:
            return orig_sampler(*args, **kwargs)
    comfy.sample.sample = pc_sample

    try:
        log.info("AITemplate detected, hijacking...")
        from custom_nodes.AIT.AITemplate.ait.load import AITLoader
        orig_apply_unet = AITLoader.apply_unet
        def apply_unet(self, *args, **kwargs):
            print("AITemplate apply_unet hijack")
            setattr(self, 'pc_applied_module', kwargs['aitemplate_module'])
            return orig_apply_unet(self, *args, **kwargs)
        AITLoader.apply_unet = apply_unet
    except Exception as e:
        log.info("AITemplate hijack failed or not necessary")
    setattr(comfy.sample.sample, 'pc_hijack_done', True)
