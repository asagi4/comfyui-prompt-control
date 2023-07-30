from . import utils as utils
from .parser import parse_prompt_schedules

from nodes import NODE_CLASS_MAPPINGS as COMFY_NODES

import logging

log = logging.getLogger("comfyui-prompt-control")


class EditableCLIPEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "filter_tag": ("STRING", {"default": ""})
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol"
    FUNCTION = "parse"

    def __init__(self):
        self.loaded_loras = {}
        self.current_loras = {}
        self.orig_clip = None

    def load_clip_lora(self, clip, loraspec):
        if not loraspec:
            return clip
        key_map = utils.get_lora_keymap(clip=clip)
        for name, params in loraspec.items():
            if name not in self.loaded_loras:
                log.warn("%s not loaded, skipping", name)
                continue
            if params['weight_clip'] == 0:
                continue
            clip = utils.load_lora(clip, self.loaded_loras[name], params['weight_clip'], key_map, clone=False)
            log.info("CLIP LoRA loaded: %s:%s", name, params['weight_clip'])
        return clip

    def do_encode(self, clip, text):
        def fallback():
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]

        # Super hacky way to call other nodes
        # <f.Nodename(param=a,param2=b)>
        if text.startswith("<f."):
            encodernode, text = text[3:].split(">", 1)
            encoderparams = {}
            paramstart = encodernode.find("(")
            paramend = encodernode.find(")")
            if paramstart > 0 and paramend > paramstart:
                ps = encodernode[paramstart + 1 : paramend]
                encodernode = encodernode[:paramstart]
                for p in ps.split(","):
                    k, v = p.split("=", 1)
                    encoderparams[k.strip().lower()] = v.strip()

            node = COMFY_NODES.get(encodernode)
            if not node or "CONDITIONING" not in node.RETURN_TYPES:
                log.error("Invalid encoder node: %s, ignoring", encodernode)
                return fallback()
            ret_index = node.RETURN_TYPES.index("CONDITIONING")
            log.info("Attempting to use %s", encodernode)
            input_types = node.INPUT_TYPES()
            r = input_types["required"]
            params = {}
            for k in r:
                t = r[k][0]
                if t == "STRING":
                    params[k] = text
                    log.info("Set %s=%s", k, params[k])
                elif t == "CLIP":
                    params[k] = clip
                    log.info("Set %s to the CLIP model", k)
                elif t in ("INT", "FLOAT"):
                    f = __builtins__[t.lower()]
                    if k in encoderparams:
                        params[k] = f(encoderparams[k])
                    else:
                        params[k] = r[k][1]["default"]
                    log.info("Set %s=%s", k, params[k])
                elif isinstance(t, list):
                    if k in encoderparams and k in t:
                        params[k] = encoderparams[k]
                    else:
                        params[k] = t[0]
                    log.info("Set %s=%s", k, params[k])
                nodefunc = getattr(node, node.FUNCTION)
            res = nodefunc(self, **params)[ret_index]
            return res
        return fallback()

    def parse(self, clip, text, filter_tag=""):
        parsed = parse_prompt_schedules(text, filter_tag.strip().upper())
        log.debug("EditableCLIPEncode schedules: %s", parsed)
        self.current_loras = {}
        self.loaded_loras = utils.load_loras_from_schedule(parsed, self.loaded_loras)
        self.orig_clip = clip.clone()
        start_pct = 0.0
        conds = []
        cond_cache = {}

        def c_str(c):
            r = [c["prompt"]]
            loras = c['loras']
            for k in sorted(loras.keys()):
                r.append(k)
                r.append(loras[k]['weight_clip'])
            return "".join(str(i) for i in r)

        for end_pct, c in parsed:
            prompt = c["prompt"]
            loras = c["loras"]
            cachekey = c_str(c)
            cond = cond_cache.get(cachekey)
            if cond is None:
                if loras != self.current_loras:
                    clip = self.load_clip_lora(self.orig_clip.clone(), loras)
                    self.current_loras = loras

                cond = self.do_encode(clip, prompt)
                cond_cache[cachekey] = cond
            else:
                pass
            # Node functions return lists of cond
            for n in cond:
                n = [n[0], n[1].copy()]
                n[1]["start_percent"] = 1.0 - start_pct
                n[1]["end_percent"] = 1.0 - end_pct
                n[1]["prompt"] = prompt
                conds.append(n)
            start_pct = end_pct
        return (conds,)
