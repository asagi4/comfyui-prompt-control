# vim: sw=4 ts=4
from __future__ import annotations

import logging
import re

from .utils import find_closing_paren, get_function, split_by_function

log = logging.getLogger("comfyui-prompt-control")


def substitute_template(template, segments, do_subs):
    def _substitute(template, segments, stack):
        name = ""
        if "$" in template:
            for name, value in sorted(segments):
                value = substitute_var(value, name, "")
                if name not in stack:
                    stack.add(name)
                    value = _substitute(value, segments, stack)
                    stack.remove(name)
                template = substitute_var(template, name, value)
        if do_subs and name not in stack:
            template = expand_subs(template)
        return template

    return _substitute(template, segments, set())


def expand_segs(text, do_subs=True):
    template, segments = split_by_function(text, "SEG", defaults=[""], require_args=True)
    named_segs = [(f.args[0].strip() or f"SEG{i + 1}", c.strip()) for i, (c, f) in enumerate(segments)]

    new_text = substitute_template(template, named_segs, do_subs).strip()
    if new_text != text.strip():
        log.debug("Template expanded to: %s", new_text)
    return new_text


def expand_subs(text):
    text, subs = get_function(text, "SUB", defaults=None)
    subs = [spec.strip() for f in subs for spec in f.args[0].split(";")]
    for spec in subs:
        if len(spec) <= 3 or spec[0] != "s":
            log.warning("Invalid SUB spec ignored: '%s'", spec)
            continue
        splitchar = spec[1]
        search, replace, *_ = spec[2:].split(splitchar)
        text = re.sub(search, replace, text)
    return text


def parse_search(search):
    arg_start = search.find("(")
    args = ""
    name = search.strip()
    if arg_start > 0:
        arg_end = find_closing_paren(search, arg_start + 1)
        if arg_end < 0:
            arg_end = len(search)
        name = search[:arg_start].strip()
        args = search[arg_start + 1 : arg_end]

    if not name:
        return None
    args = args.strip()
    # If using the form DEF(F()=$1) then the default value of $1 is the empty string
    args = [a.strip() for a in args.split(";")] if arg_start > 0 else []
    return name, args


def expand_macros(text):
    text, defs = get_function(text, "DEF", defaults=None)
    res = text
    prevres = text
    replacements = []
    for d in defs:
        if not d.args:
            continue
        r = d.args[0].split("=", 1)
        search = parse_search(r[0].strip())
        if not search or len(r) != 2:
            log.warning("Ignoring invalid DEF(%s)", d)
            continue
        replacements.append((search, r[1].strip()))
    iterations = 0
    while True:
        iterations += 1
        if iterations > 10:
            raise ValueError("Unable to resolve DEFs, make sure there are no cycles!")
        for search, replace in replacements:
            res = substitute_defcall(res, search, replace)
        if res == prevres:
            break
        prevres = res
    if res.strip() != text.strip():
        res = res.strip()
        log.debug("DEFs expanded to: %s", res)
    return res


def substitute_var(text, name, replace, boundary=r"\b"):
    if f"${name}" not in text:
        return text
    name = re.escape(str(name))
    return re.sub(rf"\${name}{boundary}", replace, text)


def substitute_defcall(text, search, replace):
    name, default_args = search
    text, defns = get_function(text, name, defaults=None, placeholder=f"DEFNCALL{name}", require_args=False)
    for i, d in enumerate(defns):
        ph = d.placeholder
        assert ph is not None, "This is a bug"
        parameters = d.args
        paramvals = []
        if parameters:
            paramvals = [x.strip() for x in parameters[0].split(";")]
        r = replace
        end_re = r"(?![0-9])"
        for i, v in enumerate(paramvals):
            r = substitute_var(r, i + 1, v, boundary=end_re)

        for i, v in enumerate(default_args):
            r = substitute_var(r, i + 1, v, boundary=end_re)

        text = text.replace(ph, r)
    return text
