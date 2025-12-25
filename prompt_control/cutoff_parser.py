from typing import TypeAlias

import lark

from .parser import flatten

cut_parser = lark.Lark(
    r"""
!start: (cut | prompt | /[][:()]/+)*
prompt: (PLAIN | WHITESPACE)+
cut: "[CUT:" prompt ":" prompt [":" NUMBER  [ ":" NUMBER [":" NUMBER [ ":" PLAIN ] ] ] ]"]"
WHITESPACE: /\s+/
PLAIN: /([^\[\]:])+/
%import common.SIGNED_NUMBER -> NUMBER
"""
)


class CutTransform(lark.Transformer):
    def __default__(self, data, children, meta):
        return children

    def NUMBER(self, args):
        return float(args)

    def cut(self, args):
        prompt, cutout, weight, strict_mask, start_from_masked, mask_token = args

        # prompts and cutouts are always sequences of str
        return (
            "".join(prompt),
            "".join(cutout),
            weight,
            strict_mask,
            start_from_masked,
            mask_token,
        )

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

    def PLAIN(self, args: str) -> str:
        return str(args)


CutResult: TypeAlias = tuple[str, str, float, float, float, str]


def parse_cuts(text: str) -> tuple[str, CutResult]:
    return CutTransform().transform(cut_parser.parse(text))
