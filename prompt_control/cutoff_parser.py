import re

from .utils import parse_args

CUTOFF_RE = re.compile(r"\[CUT:((.*?):(.*?))\]")


def noop(x):
    return x


def parse_cuts(string):
    text = CUTOFF_RE.sub(r"\2", string)
    cutoffs = CUTOFF_RE.findall(string)
    cs = []
    for x, *_ in cutoffs:
        p = x.split(":")
        args = parse_args(
            p, [(str, ""), (str, ""), (float, 0), (float, None), (float, None), (noop, None)], strip=False
        )
        args = tuple(args)
        if not args[0] or not args[1] or (args[5] is not None and not args[5].strip()):
            raise ValueError(f"Invalid CUT spec: [CUT:{x}]")
        cs.append(args)
    return text, cs
