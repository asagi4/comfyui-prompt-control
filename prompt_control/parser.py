# vim: sw=4 ts=4
import lark
import logging
from math import ceil

logging.basicConfig()
log = logging.getLogger("comfyui-prompt-control")
import re

from functools import lru_cache
from .utils import get_function, find_closing_paren

if lark.__version__ == "0.12.0":
    from sys import executable

    x = "\n".join(
        [
            "Your lark package reports an ancient version (0.12.0) and will not work. If you have the 'lark-parser' package in your Python environment, remove that and *reinstall* lark!",
            f"{executable} -m pip uninstall lark-parser lark",
            f"{executable} -m pip install lark",
        ]
    )
    log.error(x)
    raise ImportError(x)


ESCAPES = [
    ("XxPCBackslashESCAPExX", "\\"),
    ("XxPCColonESCAPExX", ":"),
    ("XxPCCommentESCAPExX", "#"),
]


def escape_specials(string):
    for ph, c in ESCAPES:
        string = string.replace(rf"\{c}", ph)
    return string


def restore_escaped(string):
    for ph, c in ESCAPES:
        string = string.replace(ph, c)
    return string


def remove_comments(string):
    r = []
    for line in string.split("\n"):
        comment = line.find("#")
        if comment > 0:
            r.append(line[:comment])
        else:
            r.append(line)
    return "\n".join(r)


prompt_parser = lark.Lark(
    r"""
!start: (prompt | /[][():|]/+)*
prompt: (emphasized | embedding | scheduled | alternate | sequence | loraspec | PLAIN | | /\\:/ | /</ | />/ | WHITESPACE)+
!emphasized: "(" prompt? ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
promptlist: ([prompt] ":")~1..3
scheduled: "[" promptlist _WS? NUMBER ["," NUMBER] "]"
        | "[" promptlist _WS? TAG "]"
sequence.5:  "[SEQ" ":" [prompt] ":" NUMBER (":" [prompt] ":" NUMBER)* "]"
alternate: "[" [prompt] ("|" [prompt])+ [":" NUMBER] "]"
loraspec.99: "<lora:" FILENAME lora_weights [lora_block_weights] ">"
lora_weights.1: (":" _WS? NUMBER)~1..2
lora_block_weights.-1: ":" PLAIN
embedding.100: "<emb:" FILENAME ">"
WHITESPACE: /\s+/
_WS: WHITESPACE
PLAIN: /([^<>\\\[\]():|]|\\.)+/
FILENAME: /[^<>:]+/
TAG: /[A-Z_]+/
%import common.SIGNED_NUMBER -> NUMBER
""",
    lexer="dynamic",
)


cut_parser = lark.Lark(
    r"""
!start: (prompt | /[][:()]/+)*
prompt: (cut | PLAIN | WHITESPACE)+
cut: "[CUT:" prompt ":" prompt [":" NUMBER  [ ":" NUMBER [":" NUMBER [ ":" PLAIN ] ] ] ]"]"
WHITESPACE: /\s+/
PLAIN: /([^\[\]:])+/
%import common.SIGNED_NUMBER -> NUMBER
"""
)


class CutTransform(lark.Transformer):
    def __default__(self, data, children, meta):
        return children

    def cut(self, args):
        prompt, cutout, weight, strict_mask, start_from_masked, mask_token = args

        return ("".join(flatten(prompt)), "".join(flatten(cutout)), weight, strict_mask, start_from_masked, mask_token)

    def start(self, args):
        prompt = []
        cuts = []
        for a in flatten(args):
            if isinstance(a, str):
                prompt.append(a)
            else:
                prompt.append(a[0])
                cuts.append(a)
        return "".join(prompt), cuts

    def PLAIN(self, args):
        return args


def parse_cuts(text):
    return CutTransform().transform(cut_parser.parse(text))


def flatten(x):
    if type(x) in [str, tuple, int, type(None)] or isinstance(x, dict) and "type" in x:
        yield x
    else:
        for g in x:
            yield from flatten(g)


def clamp(a, b, c):
    """clamp b between a and c"""
    return min(max(a, b), c)


def get_steps(tree, num_steps):
    res = [num_steps or 100]

    def tostep(s):
        steps = num_steps or 100
        if "." in str(s) or not num_steps:
            w = float(s)
            value = w * steps
        else:
            w = int(s)
            value = w

        if w > 1 and not num_steps:
            log.warning(
                "You haven't configured the number of steps for Prompt Control to use, %s will be clipped to 1.0", w
            )
            value = steps

        return int(clamp(0, value, steps))

    class CollectSteps(lark.Visitor):
        def scheduled(self, tree):
            i = tree.children[-1]
            if i and i.type == "TAG":
                return
            for i in [-1, -2]:
                if tree.children[i] is not None:
                    tree.children[i] = tostep(tree.children[i])
                    res.append(tree.children[i])

        def interp_steps(self, tree):
            tree.children[-1] = tostep(tree.children[-1] or 0.1)
            for i, _ in enumerate(tree.children[:-1]):
                tree.children[i] = tostep(tree.children[i])

            res.extend(tree.children[:-1])

        def sequence(self, tree):
            steps = tree.children[1::2]
            for i, steps in enumerate(steps):
                w = tostep(tree.children[i * 2 + 1])
                tree.children[i * 2 + 1] = w
                res.append(w)

        def alternate(self, tree):
            step_size = tostep(round(float(tree.children[-1] or 0.1), 2))
            tree.children[-1] = step_size
            res.extend([x for x in range(step_size, num_steps or 100, step_size)])

    CollectSteps().visit(tree)

    return sorted(set(res))


def at_step(step, filters, tree):
    class AtStep(lark.Transformer):
        def scheduled(self, args):
            before = None
            during = None
            after = None
            when_end = None
            pl, when, *rest = args
            if rest:
                when_end = rest[0]

            pl = list(pl)
            if len(pl) == 1:
                (during,) = pl  # [after:0.5] == [::after:0.5,0.5]
                if when_end is None:
                    when_end = when
                    after = during
            elif len(pl) == 2:
                during, after = pl  # [during:after:0.5] = [before::after:0.5,0.5]
                if when_end is None:
                    when_end = when
                    before = during
            else:
                before, during, after = pl  # [before:during:after:0.5,0.8]

            if isinstance(when, str):
                return before or "" if when not in filters else after or ""

            if when_end is None:
                when_end = 1000_000

            if step <= when:
                return before or ""
            if when < step <= when_end:
                return during or ""
            else:
                return after or ""

        def sequence(self, args):
            previous_step = 0.0
            prompts = args[::2]
            steps = args[1::2]
            for s, p in zip(steps, prompts):
                if s >= step and step >= previous_step:
                    previous_step = step
                    return p or ""
                else:
                    previous_step = s
            return ""

        def alternate(self, args):
            step_size = args[-1]
            idx = ceil(step / step_size)
            return args[(idx - 1) % (len(args) - 1)] or ""

        def start(self, args):
            prompt = []
            loraspecs = {}
            args = flatten(args)
            for a in args:
                if isinstance(a, str):
                    prompt.append(a)
                elif isinstance(a, tuple):
                    # sum identical specs together
                    n = a[0]
                    # if clip weight is not provided, use unet weight
                    w, w_clip = a[1][0], a[1][1 % len(a[1])]
                    e = loraspecs.get(n, {})
                    loraspecs[n] = {
                        "weight": round(e.get("weight", 0.0) + w, 2),
                        "weight_clip": round(e.get("weight_clip", 0.0) + w_clip, 2),
                    }
                    lbw = a[2]
                    if lbw:
                        loraspecs[n]["lbw"] = lbw
                    if loraspecs[n]["weight"] == 0 and loraspecs[n]["weight_clip"] == 0 and not lbw:
                        del loraspecs[n]
                else:
                    pass
            p = "".join(prompt)
            return {"prompt": p, "loras": loraspecs}

        def PLAIN(self, args):
            return restore_escaped(args)

        def FILENAME(self, value):
            return str(value)

        def embedding(self, args):
            return "embedding:" + str(args[0])

        def lora_weights(self, args):
            return [float(str(a)) for a in args]

        def lora_block_weights(self, args):
            vals = args[0].split(";")
            r = {}
            for v in vals:
                x = v.split("=", 2)
                if len(x) != 2:
                    continue
                k, v = x[0].strip().upper(), x[1].strip()
                r[k] = v
            return r

        def loraspec(self, args):
            name = args[0]
            params = args[1]
            lbw = args[2]

            return name, params, lbw

        def __default__(self, data, children, meta):
            for child in children:
                yield child

    return AtStep().transform(tree)


class PromptSchedule(object):
    # 0 num_steps means unconfigured
    def __init__(self, prompt, filters="", start=0.0, end=1.0, num_steps=0):
        self.filters = filters
        self.start = start
        self.end = end
        self.num_steps = num_steps
        # placeholder is restored on parse
        self.prompt = remove_comments(escape_specials(prompt.strip()))
        self.defaults = {}
        self.loaded_loras = {}

        self.parsed_prompt = self._parse(num_steps)

    def __iter__(self):
        # Filter out zero, it's only useful for interpolation
        return (x for x in self.parsed_prompt if x[0] != 0)

    def _parse(self, num_steps):
        filters = [x.strip() for x in self.filters.upper().split(",")]
        try:
            parsed = []
            tree = prompt_parser.parse(self.prompt)
            steps = get_steps(tree, num_steps=num_steps)

            def f(x):
                return round(x / (num_steps or 100), 2)

            for t in steps:
                p = at_step(t, filters, tree)
                parsed.append([f(t), p])

        except lark.exceptions.LarkError as e:
            log.error("Prompt editing parse error: %s", e)
            parsed = [[1.0, {"prompt": self.prompt, "loras": {}}]]
            raise

        # Tag filtering may return redundant prompts, so filter them out here
        res = []
        prev_end = -1

        for end_at, p in parsed:
            if end_at < self.start:
                continue
            elif end_at <= self.end:
                res.append([end_at, p])
                prev_end = end_at
            elif end_at > self.end and prev_end < self.end:
                res.append([end_at, p])
                break

        # Always use the last prompt if everything was filtered
        if len(res) == 0:
            res = [[1.0, parsed[-1][1]]]

        final = [res[0]]

        # Clean up duplicates
        for p in res[1:]:
            if p[1] != final[-1][1]:
                final.append(p)
            else:
                final[-1][0] = p[0]
        return final

    def clone(self):
        return self.with_filters()

    def with_filters(self, filters=None, start=None, end=None, defaults=None):
        def ifspecified(x, defval):
            return x if x is not None else defval

        p = PromptSchedule(
            self.prompt,
            filters=ifspecified(filters, self.filters),
            start=ifspecified(start, self.start),
            end=ifspecified(end, self.end),
            num_steps=self.num_steps,
        )
        return p

    def at_step(self, step, total_steps=1):
        _, x = self.at_step_idx(step, total_steps)
        return x

    def at_step_idx(self, step, total_steps=1):
        for i, x in enumerate(self.parsed_prompt):
            if x[0] * total_steps >= step:
                return i, x
        return len(self.parsed_prompt) - 1, self.parsed_prompt[-1]


def parse_search(search):
    arg_start = search.find("(")
    args = ""
    name = search.strip()
    if arg_start > 0:
        arg_end = find_closing_paren(search, arg_start)
        name = search[:arg_start].strip()
        args = search[arg_start + 1 : arg_end - 1]

    if not name:
        return None
    args = args.strip()
    # If using the form DEF(F()=$1) then the default value of $1 is the empty string
    if arg_start > 0:
        args = [a.strip() for a in args.split(";")]
    else:
        args = []
    return name, args


def expand_macros(text):
    text, defs = get_function(text, "DEF", defaults=None)
    res = text
    prevres = text
    replacements = []
    for d in defs:
        r = d.split("=", 1)
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


def substitute_def(text, search, replace):
    search, default_args = search
    for i, v in enumerate(default_args):
        replace = re.sub(rf"\${i+1}\b", v, replace)
    return re.sub(rf"\b{re.escape(search)}\b", replace, text)


def substitute_defcall(text, search, replace):
    name, default_args = search
    text, defns = get_function(text, name, defaults=None, placeholder=f"DEFNCALL{name}")
    for i, parameters in enumerate(defns):
        ph = f"\0DEFNCALL{name}{i}\0"
        paramvals = []
        if parameters is not None:
            paramvals = [x.strip() for x in parameters.split(";")]
        r = replace
        for i, v in enumerate(paramvals):
            r = re.sub(rf"\${i+1}\b", v, r)

        for i, v in enumerate(default_args):
            r = re.sub(rf"\${i+1}\b", v, r)

        text = text.replace(ph, r)
    return text


@lru_cache
def parse_prompt_schedules(prompt, **kwargs):
    prompt = expand_macros(prompt)
    return PromptSchedule(prompt, **kwargs)
