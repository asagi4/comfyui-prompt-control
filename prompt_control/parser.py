import lark
import logging

# Don't care about this import when testing parser
try:
    from .utils import load_loras_from_schedule
except ImportError:
    if __name__ != "__main__":
        raise


logging.basicConfig()
log = logging.getLogger("comfyui-prompt-control")

prompt_parser = lark.Lark(
    r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | sequence | loraspec | PLAIN | /</ | />/ | WHITESPACE)+
!emphasized: "(" prompt? ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] [prompt] ":" WHITESPACE? NUMBER "]"
        | "[" [prompt ":"] [prompt] ":" WHITESPACE? TAG "]"
sequence:  "[SEQ" ":" prompt ":" NUMBER (":" prompt ":" NUMBER)+ "]"
alternate: "[" prompt ("|" prompt)+ [":" NUMBER] "]"
loraspec: "<lora:" PLAIN (":" WHITESPACE? NUMBER)~1..2 ">"
WHITESPACE: /\s+/
PLAIN: /([^<>\\\[\]():|]|\\.)+/
TAG: /[A-Z_]+/
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


def get_steps(tree):
    res = [100]

    class CollectSteps(lark.Visitor):
        def scheduled(self, tree):
            i = tree.children[-1]
            if i.type == "TAG":
                return
            w = float(tree.children[-1]) * 100
            tree.children[-1] = clamp(0, w, 100)
            res.append(w)

        def sequence(self, tree):
            steps = tree.children[1::2]
            for i, steps in enumerate(steps):
                w = float(tree.children[i * 2 + 1]) * 100
                tree.children[i * 2 + 1] = clamp(0, w, 100)
                res.append(w)

        def alternate(self, tree):
            step_size = int(round(float(tree.children[-1] or 0.1), 2) * 100)
            step_size = clamp(1, step_size, 100)
            tree.children[-1] = step_size
            res.extend([x for x in range(step_size, 100, step_size)])

    CollectSteps().visit(tree)
    return sorted(set(res))


def at_step(step, filters, tree):
    class AtStep(lark.Transformer):
        def scheduled(self, args):
            before, after, when = args
            if isinstance(when, str):
                return before or () if when not in filters else after or ()

            return before or () if step <= when else after or ()

        def sequence(self, args):
            previous_step = 0.0
            prompts = args[::2]
            steps = args[1::2]
            for s, p in zip(steps, prompts):
                if s >= step and step >= previous_step:
                    previous_step = step
                    return p
                else:
                    previous_step = s
            return ()

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
                    if loraspecs[n]["weight"] == 0 and loraspecs[n]["weight_clip"] == 0:
                        del loraspecs[n]
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


class PromptSchedule(object):
    def __init__(self, prompt, filters="", start=0.0, end=1.0):
        self.filters = filters
        self.start = start
        self.end = end
        self.prompt = prompt.strip()
        self.loaded_loras = {}

        self.parsed_prompt = self._parse()

    def __iter__(self):
        return (x for x in self.parsed_prompt)

    def _parse(self):
        filters = [x.strip() for x in self.filters.upper().split(",")]
        try:
            tree = prompt_parser.parse(self.prompt)
            steps = get_steps(tree)
            parsed = [[round(t / 100, 2), at_step(t, filters, tree)] for t in steps]
        except lark.exceptions.LarkError as e:
            log.error("Prompt editing parse error: %s", e)
            parsed = [[1.0, {"prompt": self.prompt, "loras": {}}]]

        # Tag filtering may return redundant prompts, so filter them out here
        log.debug("Parse result for %s: %s", self.prompt, parsed)
        res = []
        prev_p = None
        prev_end = 0.0
        idx = 0
        for end_at, p in parsed:
            if p == prev_p:
                res[idx][0] = end_at
                continue
            if end_at <= self.start:
                continue
            elif end_at <= self.end:
                res.append([end_at, p])
                prev_end = end_at
                idx += 1
            elif end_at > self.end and prev_end < self.end:
                res.append([end_at, p])
                break
            prev_p = p

        # Always use the last prompt if everything was filtered
        if len(res) == 0:
            res = [[1.0, parsed[-1][1]]]
        # The last item always ends at 1.0
        res[-1][0] = 1.0
        log.debug("Final filtered result for %s: %s (%s, %s)", self.prompt, res, self.start, self.end)

        return res

    def with_filters(self, filters=None, start=None, end=None):
        return PromptSchedule(self.prompt, filters or self.filters, start or self.start, end or self.end)

    def at_step(self, total_steps, step):
        for x in self.parsed_prompt:
            if x[0] * total_steps >= step:
                return x
        return self.parsed_prompt[-1]

    def load_loras(self):
        self.loaded_loras = load_loras_from_schedule(self.parsed_prompt, self.loaded_loras)
        return self.loaded_loras


def parse_prompt_schedules(prompt):
    return PromptSchedule(prompt)
