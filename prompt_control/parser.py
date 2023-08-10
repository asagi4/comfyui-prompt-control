import lark
import logging
from math import ceil

logging.basicConfig()
log = logging.getLogger("comfyui-prompt-control")

prompt_parser = lark.Lark(
    r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | sequence | interpolate | loraspec | PLAIN | /</ | />/ | WHITESPACE)+
!emphasized: "(" prompt? ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] [prompt] ":" WHITESPACE? NUMBER "]"
        | "[" [prompt ":"] [prompt] ":" WHITESPACE? TAG "]"
sequence:  "[SEQ" ":" [prompt] ":" NUMBER (":" [prompt] ":" NUMBER)+ "]"
interpolate: "[INT" ":" interp_prompts ":" interp_steps "]"
interp_prompts: prompt (":" prompt)+
interp_steps: NUMBER ("," NUMBER)+ [":" NUMBER]
alternate: "[" [prompt] ("|" [prompt])+ [":" NUMBER] "]"
loraspec: "<lora:" FILENAME (":" WHITESPACE? NUMBER)~1..2 ">"
WHITESPACE: /\s+/
PLAIN: /([^<>\\\[\]():|]|\\.)+/
FILENAME: /[^<>:\/\\]+/
TAG: /[A-Z_]+/
%import common.SIGNED_NUMBER -> NUMBER
""",
    lexer="dynamic",
)


def flatten(x):
    if type(x) in [str, tuple] or isinstance(x, dict) and "type" in x:
        yield x
    else:
        for g in x:
            yield from flatten(g)


def clamp(a, b, c):
    """clamp b between a and c"""
    return min(max(a, b), c)


def get_steps(tree):
    res = [100]
    interpolation_steps = []

    def tostep(s):
        w = float(s) * 100
        w = int(clamp(0, w, 100))
        return w

    class CollectSteps(lark.Visitor):
        def scheduled(self, tree):
            i = tree.children[-1]
            if i.type == "TAG":
                return
            tree.children[-1] = tostep(tree.children[-1])
            res.append(tree.children[-1])

        def interp_steps(self, tree):
            tree.children[-1] = step = tostep(tree.children[-1] or 0.1)
            # zip start, end pairs
            for i, (start, end) in enumerate(zip(tree.children[:-1], tree.children[1:-1])):
                tree.children[i] = start = tostep(tree.children[i])
                tree.children[i + 1] = end = tostep(tree.children[i + 1])
                interpolation_steps.append((start, end, step))
                res.extend([start, end])

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

    return sorted(set(interpolation_steps)), sorted(set(res))


def at_step(step, filters, tree):
    class AtStep(lark.Transformer):
        def scheduled(self, args):
            before, after, when = args
            if isinstance(when, str):
                return before or "" if when not in filters else after or ""

            return before or "" if step <= when else after or ""

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

        def interpolate(self, args):
            prompts, starts = args
            prev_prompt = None
            if step <= starts[0]:
                return prompts[0]
            if step >= starts[-1]:
                return prompts[-1]
            for i, x in enumerate(starts[:-1]):
                if step <= x:
                    prev_prompt = prompts[i]
            return prev_prompt

        def interp_steps(self, args):
            return list(args)

        def interp_prompts(self, args):
            return ["".join(flatten(a)) for a in args]

        def alternate(self, args):
            step_size = args[-1]
            idx = ceil(step / step_size)
            return args[(idx - 1) % (len(args) - 1)] or ""

        def start(self, args):
            prompt = []
            loraspecs = {}
            args = flatten(args)
            for a in args:
                if type(a) == str:
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
                    if loraspecs[n]["weight"] == 0 and loraspecs[n]["weight_clip"] == 0:
                        del loraspecs[n]
                else:
                    pass
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
            interpolation_steps, steps = get_steps(tree)
            log.debug("Interpolation steps: %s", interpolation_steps)
            parsed = []

            def f(x):
                return round(x / 100, 2)

            for t in steps:
                p = at_step(t, filters, tree)
                interp_start = None
                interp_end = None
                interp_step = 100
                for start, end, step in interpolation_steps:
                    if t == end:
                        interp_start = max(interp_start or 0, start)
                        interp_end = min(interp_end or 100, end)
                        interp_step = min(interp_step, step)
                if interp_start is not None and interp_end is not None:
                    p["interpolations"] = (f(interp_start), f(interp_end), f(interp_step))
                parsed.append([f(t), p])

        except lark.exceptions.LarkError as e:
            log.error("Prompt editing parse error: %s", e)
            parsed = [[1.0, {"prompt": self.prompt, "loras": {}}]]

        # Tag filtering may return redundant prompts, so filter them out here
        res = []
        prev_p = None
        prev_end = 0.0

        for end_at, p in parsed:
            interpolations = p.get("interpolations")
            if p == prev_p and not interpolations:
                res[-1][0] = end_at
                continue
            if end_at <= self.start and not interpolations:
                continue
            elif end_at <= self.end:
                res.append([end_at, p])
                prev_end = end_at
            elif end_at > self.end and prev_end < self.end:
                res.append([end_at, p])
                break
            prev_p = p

        # Always use the last prompt if everything was filtered
        if len(res) == 0:
            res = [[1.0, parsed[-1][1]]]
        # The last item always ends at 1.0
        res[-1][0] = 1.0

        return res

    def with_filters(self, filters=None, start=None, end=None):
        p = PromptSchedule(self.prompt, filters or self.filters, start or self.start, end or self.end)
        p.loaded_loras = self.loaded_loras
        return p

    def at_step(self, step, total_steps=1):
        _, x = self.at_step_idx(step, total_steps)
        return x

    def at_step_idx(self, step, total_steps=1):
        for i, x in enumerate(self.parsed_prompt):
            if x[0] * total_steps >= step:
                return i, x
        return len(self.parsed_prompt) - 1, self.parsed_prompt[-1]

    def interpolation_at(self, step, total_steps=1):
        i, x = self.at_step_idx(step, total_steps)
        for y in self.parsed_prompt[i:]:
            step = min(y[0], 1.0)
            if x[1]["prompt"] != y[1]["prompt"]:
                return step, y
        return None

    def load_loras(self, lora_cache=None):
        from .utils import Timer, load_loras_from_schedule

        if lora_cache is not None:
            self.loaded_loras = lora_cache
        with Timer("PromptSchedule.load_loras()"):
            self.loaded_loras = load_loras_from_schedule(self.parsed_prompt, self.loaded_loras)
        return self.loaded_loras


def parse_prompt_schedules(prompt):
    return PromptSchedule(prompt)
