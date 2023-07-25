import lark
from .utils import getlogger

log = getlogger()

prompt_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | plain | loraspec | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" WHITESPACE? NUMBER "]"
loraspec: "<lora:" plain (":" WHITESPACE? NUMBER)~1..2 ">"
WHITESPACE: /\s+/
plain: /([^<>\\\[\]():]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")


def flatten(x):
    if type(x) in [str, tuple]:
        yield x
    else:
        for g in x:
            yield from flatten(g)

def parse_prompt_schedules(prompt):
    try:
        tree = prompt_parser.parse(prompt)
    except lark.exceptions.LarkError as e:
        log.error("Prompt editing parse error: %s", e)
        return [[1.0, {"prompt": prompt, "loras": []}]]

    # TODO: There's probably a better way to do this
    def collect(tree):
        res = [1.0]
        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                # Last element in []
                tree.children[-1] = max(min(1.0, float(tree.children[-1])), 0.0)
                res.append(tree.children[-1])

        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, when = args
                return before or () if step <= when else after

            def start(self, args):
                prompt = []
                loraspecs = []
                args = flatten(args)
                for a in args:
                    if type(a) == str:
                        prompt.append(a)
                    elif a:
                        loraspecs.append(a)
                return {"prompt": "".join(prompt), "loras": loraspecs}

            def plain(self, args):
                return args[0].value

            def loraspec(self, args):
                name = ''.join(flatten(args[0]))
                params = [float(p) for p in args[1:]]
                return name, params

            def __default__(self, data, children, meta):
                for child in children:
                    yield child

        return AtStep().transform(tree)

    parsed = [[t, at_step(t, tree)] for t in collect(tree)]
    return parsed
