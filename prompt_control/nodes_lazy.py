import logging
from .parser import parse_prompt_schedules
from comfy_execution.graph_utils import GraphBuilder, is_link

from .prompts import get_function

log = logging.getLogger("comfyui-prompt-control")

from .utils import consolidate_schedule, find_nonscheduled_loras


def cache_key_hack(inputs):
    out = inputs.copy()
    if not is_link(inputs["text"]):
        out["text"] = cache_key_from_inputs(**inputs)
    return out


def create_lora_loader_nodes(graph, model, clip, loras):
    for path, info in loras.items():
        log.info("Creating LoraLoader for %s", path)
        loader = graph.node("LoraLoader")
        loader.set_input("model", model)
        loader.set_input("clip", clip)
        loader.set_input("strength_model", info["weight"])
        loader.set_input("strength_clip", info["weight_clip"])
        loader.set_input("lora_name", path)
        model = loader.out(0)
        clip = loader.out(1)
    return model, clip


def create_hook_nodes_for_lora(graph, path, info, existing_node, start_pct, end_pct):
    prev_keyframe = None
    next_keyframe = None
    if not existing_node:
        log.debug("Creating hook for %s", path)
        hook_node = graph.node("CreateHookLora")
        hook_node.set_input("lora_name", path)
        hook_node.set_input("strength_model", info["weight"])
        hook_node.set_input("strength_clip", info["weight_clip"])
        prev_hook_kf = None
        if start_pct > 0:
            log.debug("Creating KF (0, %s) for %s", start_pct, path)
            prev_keyframe = graph.node("CreateHookKeyframe")
            prev_keyframe.set_input("strength_mult", 0.0)
            prev_keyframe.set_input("start_percent", 0.0)
            prev_hook_kf = prev_keyframe.out(0)
    else:
        log.debug("Hook already created for %s", path)
        hook_node, prev_keyframe = existing_node
        prev_hook_kf = prev_keyframe.out(0)

    if (
        prev_keyframe
        and prev_keyframe.get_input("start_pct") == start_pct
        and prev_keyframe.get_input("strength_mult") == 0.0
    ):
        next_keyframe = prev_keyframe
        log.debug("Previous keyframe for %s starts at %s and has 0 strength, overriding", path, start_pct)
    else:
        log.debug("Creating keyframe for %s, start=%s ", path, start_pct)
        next_keyframe = graph.node("CreateHookKeyframe")
        next_keyframe.set_input("start_percent", start_pct)
        next_keyframe.set_input("prev_hook_kf", prev_hook_kf)

    next_keyframe.set_input("strength_mult", 1.0)
    prev_hook_kf = next_keyframe.out(0)
    if end_pct < 1.0:
        log.debug("Creating end keyframe for %s, start=%s", path, start_pct)
        next_keyframe = graph.node("CreateHookKeyframe")
        next_keyframe.set_input("strength_mult", 0.0)
        next_keyframe.set_input("start_percent", end_pct)
        next_keyframe.set_input("prev_hook_kf", prev_hook_kf)
    return hook_node, next_keyframe


def build_lora_schedule(graph, schedule, model, clip, apply_hooks=True, return_hooks=True):
    # This gets rid of non-existent LoRAs
    consolidated = consolidate_schedule(schedule)
    non_scheduled = find_nonscheduled_loras(consolidated)
    model, clip = create_lora_loader_nodes(graph, model, clip, non_scheduled)

    hook_nodes = {}
    start_pct = 0.0

    def key(lora, info):
        return f"{lora}-{info['weight']}-{info['weight_clip']}"

    for end_pct, loras in consolidated:
        for lora, info in loras.items():
            if non_scheduled.get(lora) == info:
                continue
            k = key(lora, info)
            existing_node = hook_nodes.get(k)
            hook_nodes[k] = create_hook_nodes_for_lora(graph, lora, info, existing_node, start_pct, end_pct)
        start_pct = end_pct

    hooks = []
    # Attach the keyframe chain to the hook node
    for hook, kfs in hook_nodes.values():
        n = graph.node("SetHookKeyframes")
        n.set_input("hooks", hook.out(0))
        n.set_input("hook_kf", kfs.out(0))
        hooks.append(n)

    res = None
    # Finally, combine all hooks and optionally apply
    if len(hooks) > 0:
        res = hooks[0]
        for h in hooks[:1]:
            n = graph.node("CombineHooks2")
            n.set_input("hooks_A", res.out(0))
            n.set_input("hooks_B", h.out(0))
            res = n
        res = res.out(0)
        if apply_hooks:
            n = graph.node("SetClipHooks")
            n.set_input("clip", clip)
            n.set_input("hooks", res)
            n.set_input("apply_to_conds", True)
            n.set_input("schedule_clip", True)
            clip = n.out(0)

    r = graph.finalize()

    if return_hooks:
        ret = (model, clip, res)
    else:
        ret = (model, clip)

    return {"result": ret, "expand": r}


class PCLazyLoraLoaderAdvanced:
    CACHE_KEY = cache_key_hack

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "model": ("MODEL", {"rawLink": True}),
                "clip": ("CLIP", {"rawLink": True}),
            },
            "optional": {
                "apply_hooks": ("BOOLEAN", {"default": True}),
                "tags": ("STRING", {"default": ""}),
                "start": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.01}),
                "end": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.01}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "HOOKS")
    OUTPUT_TOOLTIPS = ("Returns a model and clip with LoRAs scheduled",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, model, clip, text, unique_id, apply_hooks=True, tags="", start=0.0, end=1.0):
        schedule = parse_prompt_schedules(text, filters=tags, start=start, end=end)
        graph = GraphBuilder(f"PCLazyLoraLoaderAdvanced-{unique_id}")
        return build_lora_schedule(graph, schedule, model, clip, apply_hooks=apply_hooks, return_hooks=True)


class PCLazyLoraLoader:
    CACHE_KEY = cache_key_hack

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"rawLink": True}),
                "clip": ("CLIP", {"rawLink": True}),
                "text": ("STRING", {"multiline": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
    )
    OUTPUT_TOOLTIPS = ("Returns a model and clip with LoRAs scheduled",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, model, clip, text, unique_id):
        graph = GraphBuilder(f"PCLazyLoraLoader-{unique_id}")
        schedule = parse_prompt_schedules(text)
        return build_lora_schedule(graph, schedule, model, clip, apply_hooks=True, return_hooks=False)


def build_scheduled_prompts(graph, schedules, clip):
    nodes = []
    start_pct = 0.0
    prompt_cache = {}
    for end_pct, c in schedules:
        p = c["prompt"]
        p, classnames = get_function(p, "NODE", ["PCTextEncode", "text"])
        classname = "PCTextEncode"
        paramname = "text"
        if classnames:
            classname = classnames[0][0]
            paramname = classnames[0][1]
        node = prompt_cache.get((p, classname, paramname))
        if not node:
            node = graph.node(classname)
            node.set_input("clip", clip)
            node.set_input(paramname, p)
            prompt_cache[(p, classname, paramname)] = node
        timestep = graph.node("ConditioningSetTimestepRange")
        timestep.set_input("conditioning", node.out(0))
        timestep.set_input("start", start_pct)
        timestep.set_input("end", end_pct)
        nodes.append(timestep)
        start_pct = end_pct
    node = nodes[0]
    for othernode in nodes[1:]:
        combiner = graph.node("ConditioningCombine")
        combiner.set_input("conditioning_1", node.out(0))
        combiner.set_input("conditioning_2", othernode.out(0))
        node = combiner

    g = graph.finalize()

    return {"result": (node.out(0),), "expand": g}


def cache_key_from_inputs(text, tags="", start=0.0, end=1.0, **kwargs):
    schedules = parse_prompt_schedules(text, filters=tags, start=start, end=end)
    return [(pct, s["prompt"]) for pct, s in schedules]


class PCLazyTextEncode:
    CACHE_KEY = cache_key_hack

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP", {"rawLink": True}), "text": ("STRING", {"multiline": True})},
        }

    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("A fully encoded and scheduled conditioning",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, clip, text):
        schedules = parse_prompt_schedules(text)
        graph = GraphBuilder()
        return build_scheduled_prompts(graph, schedules, clip)


class PCLazyTextEncodeAdvanced:
    CACHE_KEY = cache_key_hack

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP", {"rawLink": True}), "text": ("STRING", {"multiline": True})},
            "optional": {
                "tags": ("STRING", {"default": ""}),
                "start": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.01}),
                "end": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.01}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "promptcontrol"
    FUNCTION = "apply"

    def apply(self, clip, text, unique_id, tags="", start=0.1, end=1.0):
        schedules = parse_prompt_schedules(text, filters=tags, start=start, end=end)
        graph = GraphBuilder(f"PCLazyTextEncodeAdvanced-{unique_id}")
        return build_scheduled_prompts(graph, schedules, clip)


NODE_CLASS_MAPPINGS = {
    "PCLazyTextEncode": PCLazyTextEncode,
    "PCLazyTextEncodeAdvanced": PCLazyTextEncodeAdvanced,
    "PCLazyLoraLoader": PCLazyLoraLoader,
    "PCLazyLoraLoaderAdvanced": PCLazyLoraLoaderAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PCLazyTextEncode": "PC: Schedule Prompt",
    "PCLazyTextEncodeAdvanced": "PC: Schedule prompt (Advanced)",
    "PCLazyLoraLoader": "PC: Schedule LoRAs",
    "PCLazyLoraLoaderAdvanced": "PC: Schedule LoRAs (Advanced)",
}
