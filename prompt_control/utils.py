from pathlib import Path
import re
import logging

# Allow testing
if __name__ != "__main__":
    import folder_paths

log = logging.getLogger("comfyui-prompt-control")


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


def get_function(text, func, defaults, return_func_name=False):
    rex = re.compile(rf"\b{func}\(", re.MULTILINE)
    instances = []
    match = rex.search(text)
    while match:
        # Match start, content start
        start, after_first_paren = match.span()
        funcname = text[start : after_first_paren - 1]
        end = find_closing_paren(text, after_first_paren)
        args = parse_strings(text[after_first_paren:end], defaults)
        if return_func_name:
            instances.append((funcname, args))
        else:
            instances.append(args)

        text = text[:start] + text[end + 1 :]
        match = rex.search(text)
    return text, instances


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
    filenames = folder_paths.get_filename_list("loras")
    # Return exact matches as is
    if name in filenames:
        return name
    # Some autocompletion scripts replace _ with spaces
    for n in [name, name.replace(" ", "_")]:
        for f in filenames:
            p = Path(f).with_suffix("")
            if p.name == n or str(p) == n:
                return f
    return None
