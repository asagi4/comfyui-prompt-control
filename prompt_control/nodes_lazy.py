import logging
from .parser import parse_prompt_schedules
from comfy_execution.graph_utils import GraphBuilder

log = logging.getLogger("comfyui-prompt-control")


class PCEncodeLazy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"clip": ("CLIP",), "prompt": ("STRING", {"multiline": True})},
            "optional": {"defaults": ("SCHEDULE_DEFAULTS",)},
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("A fully encoded and scheduled conditioning",)
    CATEGORY = "promptcontrol/_unstable"
    FUNCTION = "apply"

    def apply(self, clip, prompt, dynprompt, unique_id, defaults=None):
        schedules = parse_prompt_schedules(prompt)
        graph = GraphBuilder(f"PCEncodeLazy-{unique_id}")

        this_node = dynprompt.get_node(unique_id)
        print("Lazy", this_node)

        nodes = []
        start_pct = 0.0
        for end_pct, c in schedules:
            p = c["prompt"]
            node = graph.node("PCEncodeSingle")
            timestep = graph.node("ConditioningSetTimestepRange")
            node.set_input("clip", this_node["inputs"]["clip"])
            node.set_input("prompt", p)
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

        return {"result": (node.out(0),), "expand": graph.finalize()}


NODE_CLASS_MAPPINGS = {
    "PCEncodeLazy": PCEncodeLazy,
}

NODE_DISPLAY_NAME_MAPPINGS = {"PCEncodeLazy": "PromptControl Encode (Lazy) (EXPERIMENTAL)"}
