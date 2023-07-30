import lark

try:
    from jinja2 import Template

    def expand_template(string):
        for x in ["<%", "<=", "<#", "#>", "=>", "%>"]:
            if x in string:
                return Template(
                    string,
                    block_start_string="<%",
                    block_end_string="%>",
                    variable_start_string="<=",
                    variable_end_string="=>",
                    comment_start_string="<#",
                    comment_end_string="#>",
                ).render()
        return string

except ImportError:
    print("Failed to import jinja2")

    def expand_template(string):
        return string


import logging

logging.basicConfig()
log = logging.getLogger("comfyui-prompt-control")

prompt_parser = lark.Lark(
    r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | loraspec | PLAIN | /</ | />/ | WHITESPACE)+
!emphasized: "(" prompt? ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" WHITESPACE? NUMBER "]"
alternate: "[" prompt ("|" prompt)+ [":" NUMBER] "]"
loraspec: "<lora:" PLAIN (":" WHITESPACE? NUMBER)~1..2 ">"
WHITESPACE: /\s+/
PLAIN: /([^<>\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""",
    lexer="dynamic",
)


def flatten(x):
    if type(x) in [str, tuple]:
        yield x
    else:
        for g in x:
            yield from flatten(g)


def clamp(a, b, c):
    """clamp b between a and c"""
    return min(max(a, b), c)


def parse_prompt_schedules(prompt):
    prompt = expand_template(prompt)
    prompt = prompt.strip()
    log.debug("Parsing: %s", prompt)

    try:
        tree = prompt_parser.parse(prompt)
    except lark.exceptions.LarkError as e:
        log.error("Prompt editing parse error: %s", e)
        return [[1.0, {"prompt": prompt, "loras": {}}]]

    # TODO: There's probably a better way to do this
    # Use 100 instead of floats here to make math easier
    # Convert to percentages later
    def collect(tree):
        res = [100]

        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                # Last element in []
                w = float(tree.children[-1]) * 100
                tree.children[-1] = clamp(0, w, 100)
                res.append(w)

            def alternate(self, tree):
                step_size = int(round(float(tree.children[-1] or 0.1), 2) * 100)
                step_size = clamp(1, step_size, 100)
                tree.children[-1] = step_size
                res.extend([x for x in range(step_size, 100, step_size)])

        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, when = args
                return before or () if step <= when else after

            def alternate(self, args):
                step_size = args[-1]
                idx = int(step / step_size)
                return args[(idx - 1) % (len(args) - 1)]

            def start(self, args):
                prompt = []
                loraspecs = {}
                args = flatten(args)
                for a in args:
                    if type(a) == str:
                        prompt.append(a)
                    elif a:
                        # sum identical specs together
                        n = a[0]
                        # if clip weight is not provided, use unet weight
                        w, w_clip = a[1][0], a[1][1 % len(a[1])]
                        e = loraspecs.get(n, {})
                        loraspecs[n] = {
                            "weight": round(e.get("weight", 0.0) + w, 2),
                            "weight_clip": round(e.get("weight_clip", 0.0) + w_clip, 2),
                        }
                p = "".join(prompt)
                return {"prompt": p, "loras": loraspecs}

            def plain(self, args):
                return args[0].value

            def loraspec(self, args):
                name = "".join(flatten(args[0]))
                params = [float(p) for p in args[1:]]
                return name, params

            def __default__(self, data, children, meta):
                for child in children:
                    yield child

        return AtStep().transform(tree)

    parsed = [[round(t / 100, 2), at_step(t, tree)] for t in collect(tree)]
    return parsed
