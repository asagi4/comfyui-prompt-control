from __future__ import annotations
from pathlib import Path
import re
import logging
import copy

from dataclasses import dataclass
from typing import Any, TypeAlias, Iterator, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # flakes8: noqa

FunctionArgs: TypeAlias = list[str]
ComfyConditioning: TypeAlias = tuple["torch.Tensor", dict[str, Any]]


@dataclass
class FunctionSpec:
    name: str
    args: FunctionArgs
    position: int
    placeholder: str | None


# Allow testing
try:
    from folder_paths import get_filename_list
except ImportError:

    def get_filename_list(folder_name) -> list[str]:
        return []


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


def smarter_split(separator: str, string: str) -> list[str]:
    """Does not break () when splitting"""
    splits = []
    prev = 0
    idx = 0
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


def find_closing_paren(text: str, start: int) -> int:
    stack = 1
    for i, char in enumerate(text[start:]):
        if char == ")":
            stack -= 1
        elif char == "(":
            stack += 1
        if stack == 0:
            return start + i
    return -1


def find_function_spans(
    text: str, func: str, require_args: bool, defaults: FunctionArgs | None
) -> Iterator[tuple[int, int, str, FunctionArgs]]:
    if require_args:
        rex = re.compile(rf"\b{func}\(", re.MULTILINE)
    else:
        rex = re.compile(rf"\b{func}\b", re.MULTILINE)

    idx = 0
    match = rex.search(text)
    while match:
        start, at_paren = match.span()
        if require_args:
            at_paren = at_paren - 1
        funcname = text[start:at_paren]
        after_first_paren = at_paren + 1
        if text[at_paren:after_first_paren] == "(":
            end = find_closing_paren(text, after_first_paren)
            if end < 0:
                continue
            args = parse_strings(text[after_first_paren:end], defaults)
            end += 1
        else:
            end = at_paren
            args = defaults or []
        yield idx + start, idx + end, funcname, args
        idx = idx + end
        text = text[end:]
        match = rex.search(text)


def get_function(
    text: str, func: str, defaults: list[str] | None, placeholder: str = "", require_args: bool = True
) -> tuple[str, list[FunctionSpec]]:
    spans = [x.span() for x in re.finditer(r'".+?"', text)]
    instances = []
    count = 0
    chunks = []
    current = 0
    for start, end, funcname, args in find_function_spans(text, func, require_args, defaults):
        ph = None
        if spans_include(spans, start, end):
            continue
        if placeholder:
            ph = f"\0{placeholder}{count}\0"
        instances.append(FunctionSpec(funcname, args, start, ph))
        chunks.append(text[current:start] + (ph or ""))
        current = end
        count += 1
    chunks.append(text[current:])
    text = "".join(chunks)
    return text, instances


def spans_include(spans: list[tuple[int, int]], s: int, e: int) -> bool:
    return any((s > a and e < b) for a, b in spans)


def split_quotable(text: str, regexp: str) -> Iterator[str]:
    start_from = 0
    spans = [x.span() for x in re.finditer(r'".+?"', text)]
    for x in re.finditer(regexp, text):
        s, e = x.span()
        if not spans_include(spans, s, e):
            yield text[start_from:s].strip()
            start_from = e
    yield text[start_from:].strip()


def split_by_function(
    text: str, func: str, defaults: list[str] | None = None, require_args: bool = True
) -> tuple[str, list[tuple[str, FunctionSpec]]]:
    """
    Splits a string by function calls, returning the leftover text along with a list of functions with their associated text chunk.
    """
    text, functions = get_function(text, func, defaults, require_args=require_args)
    chunks = []
    prev = 0
    for f in functions:
        chunks.append(text[prev : f.position])
        prev = f.position
    chunks.append(text[prev:])
    r = []
    for i, f in enumerate(functions):
        r.append((chunks[i + 1], f))
    return chunks[0], r


T = TypeVar("T")


def parse_args(strings: list[str], arg_spec: list[tuple[Any, T]], strip: bool = True) -> list[T]:
    args = [s[1] for s in arg_spec]
    for i, spec in list(enumerate(arg_spec))[: len(strings)]:
        try:
            if strip:
                strings[i] = strings[i].strip()
            f = spec[0]
            args[i] = f(strings[i])
        except ValueError:
            pass
    return args


def parse_floats(string: str, defaults: list[float], split_re: str = ",") -> list[float]:
    spec = [(float, d) for d in defaults]
    return parse_args(re.split(split_re, string.strip()), spec)


def parse_strings(
    string: str, defaults: FunctionArgs | None, split_re: str = r"(?<!\\),", replace: tuple[str, str] = (r"\,", ",")
) -> FunctionArgs:
    if defaults is None:
        return [string]
    spec = [(str, d) for d in defaults]
    splits = re.split(split_re, string)
    if replace:
        f, t = replace
        splits = [s.replace(f, t) for s in splits]
    return parse_args(splits, spec, strip=False)


def safe_float(f: Any, default: float) -> float:
    if f is None:
        return default
    try:
        return round(float(f), 2)
    except ValueError:
        return default


def lora_name_to_file(name: str) -> str | None:
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
