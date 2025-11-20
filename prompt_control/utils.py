from pathlib import Path
import re
import logging
import copy

# Allow testing
try:
    from folder_paths import get_filename_list
except ImportError:

    def get_filename_list(x):
        raise NotImplementedError("How did you get here?")


log = logging.getLogger("comfyui-prompt-control")


def call_node(cls, *args, **kwargs):
    if hasattr(cls, "execute"):
        # v3 node
        return cls.execute(*args, **kwargs)
    else:
        func = getattr(cls(), cls.FUNCTION)
        return func(*args, **kwargs)


def consolidate_schedule(prompt_schedule):
    prev_loras = {}
    not_found = []
    consolidated = []
    for end_pct, c in reversed(list(prompt_schedule)):
        loras = {}
        for k, v in c["loras"].items():
            if k in not_found:
                continue
            path = lora_name_to_file(k)
            if path is None:
                not_found.append(k)
                continue
            loras[path] = v

        if loras != prev_loras:
            consolidated.append((end_pct, loras))
        prev_loras = loras
    for k in not_found:
        log.warning("LoRA '%s' not found, ignoring...", k)
    return list(reversed(consolidated))


def find_nonscheduled_loras(consolidated_schedule):
    consolidated_schedule = list(consolidated_schedule)
    if not consolidated_schedule:
        return {}
    last_end, candidate_loras = consolidated_schedule[0]
    to_remove = set()
    for candidate, weights in candidate_loras.items():
        for end, loras in consolidated_schedule[1:]:
            last_end = end
            if loras.get(candidate) != weights:
                to_remove.add(candidate)
    # No candidates if the schedule does not span full time
    if last_end < 1.0:
        return {}
    return {k: v for (k, v) in candidate_loras.items() if k not in to_remove}


def smarter_split(separator, string):
    """Does not break () when splitting"""
    splits = []
    prev = 0
    stack = 0
    escape = False
    for idx, x in enumerate(string):
        if x == "(" and not escape:
            stack += 1
        elif x == ")" and not escape:
            stack = max(0, stack - 1)
        elif x == separator and stack == 0:
            splits.append(string[prev:idx])
            prev = idx + 1
        escape = x == "\\"

    splits.append(string[prev : idx + 1])
    return splits


def find_closing_paren(text, start):
    stack = 1
    for i, char in enumerate(text[start:]):
        if char == ")":
            stack -= 1
        elif char == "(":
            stack += 1
        if stack == 0:
            return start + i
    # Implicit closing paren after end
    return len(text)


def get_function(text, func, defaults, return_func_name=False, placeholder="", return_dict=False, require_args=True):
    if require_args:
        rex = re.compile(rf"\b{func}\(", re.MULTILINE)
    else:
        rex = re.compile(rf"\b{func}\b", re.MULTILINE)
    instances = []
    match = rex.search(text)
    count = 0
    while match:
        # Match start, content start
        start, at_paren = match.span()
        if require_args:
            at_paren = at_paren - 1
        funcname = text[start:at_paren]
        after_first_paren = at_paren + 1
        if text[at_paren:after_first_paren] == "(":
            end = find_closing_paren(text, after_first_paren)
            args = parse_strings(text[after_first_paren:end], defaults)
            end += 1
        else:
            end = at_paren
            args = defaults
        ph = None
        if placeholder:
            ph = f"\0{placeholder}{count}\0"
        if return_dict:
            instances.append(
                {
                    "name": funcname,
                    "args": args,
                    "position": start,
                    "placeholder": ph,
                }
            )
        elif return_func_name:
            instances.append((funcname, args))
        else:
            instances.append(args)

        if placeholder:
            text = text[:start] + f"\0{placeholder}{count}\0" + text[end:]
        else:
            text = text[:start] + text[end:]
        match = rex.search(text)
        count += 1
    return text, instances


def split_by_function(text, func, defaults=None, require_args=True):
    """
    Splits a string by function calls, returning the text preceding the first call and a list of dictionaries with a "text" key with the prompt before the next split or until hthe end of the text.
    """
    text, functions = get_function(text, func, defaults, return_dict=True, require_args=require_args)
    chunks = []
    prev = 0
    for f in functions:
        chunks.append(text[prev : f["position"]])
        prev = f["position"]
    chunks.append(text[prev:])
    for i, f in enumerate(functions):
        f["text"] = chunks[i + 1]
    return chunks[0], functions


def parse_args(strings, arg_spec, strip=True):
    args = [s[1] for s in arg_spec]
    for i, spec in list(enumerate(arg_spec))[: len(strings)]:
        try:
            if strip:
                strings[i] = strings[i].strip()
            args[i] = spec[0](strings[i])
        except ValueError:
            pass
    return args


def parse_floats(string, defaults, split_re=","):
    spec = [(float, d) for d in defaults]
    return parse_args(re.split(split_re, string.strip()), spec)


def parse_strings(string, defaults, split_re=r"(?<!\\),", replace=(r"\,", ",")):
    if defaults is None:
        return string
    spec = [(lambda x: x, d) for d in defaults]
    splits = re.split(split_re, string)
    if replace:
        f, t = replace
        splits = [s.replace(f, t) for s in splits]
    return parse_args(splits, spec, strip=False)


def safe_float(f, default):
    if f is None:
        return default
    try:
        return round(float(f), 2)
    except ValueError:
        return default


def lora_name_to_file(name):
    filenames = get_filename_list("loras")
    # Return exact matches as is
    if name in filenames:
        return name
    # Some autocompletion scripts replace _ with spaces
    for n in [name, name.replace(" ", "_")]:
        for f in filenames:
            p = Path(f).with_suffix("")
            if p.name == n or str(p) == n:
                return f
    # Finally, try to find unique match from parts
    parts = name.split()
    search = [f for f in filenames if all(p in f for p in parts)]
    if len(search) == 1:
        return search[0]

    return None


def map_inputs(input_map, inputs):
    new_inputs = {}
    for k in inputs:
        key = inputs[k]
        new_inputs[k] = key
        if isinstance(key, list):
            key = tuple(key)
            x = input_map.get(key, inputs[k])
            new_inputs[k] = x
    return new_inputs


def expand_graph(node_mappings, graph):
    input_map = {}
    new_graph = copy.deepcopy(graph)
    for k in graph:
        data = graph[k]
        if not isinstance(data, dict) or "class_type" not in data or data["class_type"] not in node_mappings:
            continue
        node = node_mappings[data["class_type"]]()
        inputs = map_inputs(input_map, data["inputs"].copy())
        inputs["unique_id"] = k
        fn = getattr(node, getattr(node, "FUNCTION"))
        expansion = fn(**inputs)
        for i, v in enumerate(expansion["result"]):
            input_map[(k, i)] = v
        del new_graph[k]
        new_graph.update(expansion["expand"])

    for k in new_graph:
        data = new_graph[k]
        data["inputs"] = map_inputs(input_map, data["inputs"])
    return new_graph
