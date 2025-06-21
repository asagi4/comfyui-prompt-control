#!/usr/bin/env python3
from prompt_control.utils import expand_graph
from prompt_control.nodes_lazy import NODE_CLASS_MAPPINGS as LN
import json
import sys


# Needs ComfyUI in Python path
# Usage: PYTHONPATH=../..:. python tools/expand_graph < graph_in_api_format.json > out.json
if __name__ == "__main__":
    graph = json.load(sys.stdin)
    new = expand_graph(LN, graph)
    print(json.dumps(new))
