# vim: sw=4 ts=4
from __future__ import annotations

import logging
import re

from .utils import find_closing_paren, get_function

logging.basicConfig()
log = logging.getLogger("comfyui-prompt-control")


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
            return text
        for search, replace in replacements:
            res = substitute_defcall(res, search, replace)
        if res == prevres:
            break
        prevres = res
    if res.strip() != text.strip():
        res = res.strip()
        log.info("DEFs expanded to: %s", res)
    return res


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
        for i, v in enumerate(paramvals):
            r = re.sub(rf"\${i + 1}\b", v, r)

        for i, v in enumerate(default_args):
            r = re.sub(rf"\${i + 1}\b", v, r)

        text = text.replace(ph, r)
    return text
