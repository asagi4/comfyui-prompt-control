from pathlib import Path
import re
import logging

import folder_paths

log = logging.getLogger("comfyui-prompt-control-legacy")


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
