import comfy_execution.caching
from comfy_execution.graph_utils import is_link
import nodes
from os import environ

import logging

log = logging.getLogger("comfyui-prompt-control")


include_unique_id_in_input = comfy_execution.caching.include_unique_id_in_input


def promptcontrol_get_immediate_node_signature(self, dynprompt, node_id, ancestor_order_mapping):
    if not dynprompt.has_node(node_id):
        # This node doesn't exist -- we can't cache it.
        return [float("NaN")]
    node = dynprompt.get_node(node_id)
    class_type = node["class_type"]
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    inputs = node["inputs"]
    if hasattr(class_def, "CACHE_KEY"):
        inputs = getattr(class_def, "CACHE_KEY")(inputs)
    signature = [class_type, self.is_changed_cache.get(node_id)]
    if (
        self.include_node_id_in_input()
        or (hasattr(class_def, "NOT_IDEMPOTENT") and class_def.NOT_IDEMPOTENT)
        or include_unique_id_in_input(class_type)
    ):
        signature.append(node_id)
    for key in sorted(inputs.keys()):
        if is_link(inputs[key]):
            (ancestor_id, ancestor_socket) = inputs[key]
            ancestor_index = ancestor_order_mapping[ancestor_id]
            signature.append((key, ("ANCESTOR", ancestor_index, ancestor_socket)))
        else:
            signature.append((key, inputs[key]))
    return signature


def init():
    if environ.get("PROMPTCONTROL_ENABLE_CACHE_HACK") != "1":
        return
    log.warning("Enabling Prompt Control cache hack")
    comfy_execution.caching.CacheKeySetInputSignature.get_immediate_node_signature = (
        promptcontrol_get_immediate_node_signature
    )
