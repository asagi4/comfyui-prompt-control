import lark
import logging
from math import ceil

logging.basicConfig()
log = logging.getLogger("comfyui-prompt-control")

prompt_parser = lark.Lark(
    r"""
!start: (prompt | /[][():|]/+)*
prompt: (emphasized | embedding | scheduled | alternate | sequence | interpolate | loraspec | PLAIN | /</ | />/ | WHITESPACE)+
!emphasized: "(" prompt? ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] [prompt] ":" _WS? NUMBER ["," NUMBER] "]"
        | "[" [prompt ":"] [prompt] ":" _WS? TAG "]"
sequence:  "[SEQ" ":" [prompt] ":" NUMBER (":" [prompt] ":" NUMBER)+ "]"
interpolate.100: "[INT" ":" interp_prompts ":" interp_steps "]"
interp_prompts: prompt (":" [prompt])+
interp_steps: NUMBER ("," NUMBER)+ [":" NUMBER]
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

            interpolation_steps.append((tuple(tree.children[:-1]), tree.children[-1]))
            res.extend(tree.children[:-1])

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
            when_end = None
            before, after, when, *rest = args
            if isinstance(when, str):
                return before or "" if when not in filters else after or ""

            if rest:
                when_end = rest[0]

            if when_end is not None and step <= when and before is not None:
                return ""

            if when_end is not None and (step > when and step <= when_end):
                # handle [a:0,1]
                if before is None:
                    return after or ""
                return before or ""

            if when_end is not None and step >= when_end:
                # handle [a:0,1]
                if before is None:
                    return ""
                return after or ""

            if step <= when:
                return before or ""
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

        def interpolate(self, args):
            prompts, starts = args
            starts = starts[:-1]
            prev_prompt = None
            if step < starts[0]:
                return prompts[0]
            for i, x in enumerate(starts):
                prev_prompt = prompts[i]
                if x >= step:
                    break
            return prev_prompt

        def interp_steps(self, args):
            return list(args)

        def interp_prompts(self, args):
            return ["".join(flatten(a or [])) for a in args]

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
            return args.replace("\\:", ":")

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
    def __init__(self, prompt, filters="", start=0.0, end=1.0, defaults=None, masks=None):
        self.filters = filters
        self.start = start
        self.end = end
        self.prompt = prompt.strip()
        self.defaults = {}
        if defaults:
            self.defaults = defaults
        self.loaded_loras = {}

        self.interpolations = None
        self.parsed_prompt = None
        self.interpolations, self.parsed_prompt = self._parse()
        self.masks = masks
        if masks is None:
            self.masks = []

    def __iter__(self):
        # Filter out zero, it's only useful for interpolation
        return (x for x in self.parsed_prompt if x[0] != 0)

    def _parse(self):
        filters = [x.strip() for x in self.filters.upper().split(",")]
        try:
            parsed = []
            interpolations = set()
            tree = prompt_parser.parse(self.prompt)
            interpolation_steps, steps = get_steps(tree)
            log.debug("Interpolation steps: %s", interpolation_steps)

            def f(x):
                return round(x / 100, 2)

            for t in steps:
                p = at_step(t, filters, tree)
                for control_points, step in interpolation_steps:
                    interp_start = None
                    interp_end = None
                    if t == control_points[-1]:
                        interp_start = max(control_points[0], int(self.start * 100))
                        interp_end = min(control_points[-1], int(self.end * 100))
                        control_points = tuple(
                            sorted(set(f(c) for c in control_points if c >= interp_start or c <= interp_end))
                        )
                    if interp_start is not None and interp_end is not None and interp_end > interp_start:
                        interpolations.add((control_points, f(step)))
                parsed.append([f(t), p])

        except lark.exceptions.LarkError as e:
            log.error("Prompt editing parse error: %s", e)
            parsed = [[1.0, {"prompt": self.prompt, "loras": {}}]]

        # Tag filtering may return redundant prompts, so filter them out here
        res = []
        prev_p = None
        prev_end = -1

        for end_at, p in parsed:
            # Preserve prompt if it ends at the start of an interpolation, otherwise bump its end time
            if p == prev_p and res[-1][0] not in [x[0][0] for x in interpolations]:
                res[-1][0] = end_at
                continue
            if end_at < self.start:
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

        return interpolations, res

    def add_masks(self, *masks):
        for mask in masks:
            if mask is not None:
                self.masks.append(mask)

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
            defaults=ifspecified(defaults, self.defaults),
            masks=self.masks[:],
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

    def interpolation_at(self, step, total_steps=1):
        i, x = self.at_step_idx(step, total_steps)
        for y in self.parsed_prompt[i:]:
            step = min(y[0], 1.0)
            if x[1]["prompt"] != y[1]["prompt"]:
                return step, y
        return 1.0, self.parsed_prompt[-1]

    def load_loras(self, lora_cache=None):
        from .utils import Timer, load_loras_from_schedule

        if lora_cache is not None:
            self.loaded_loras = lora_cache
        with Timer("PromptSchedule.load_loras()"):
            self.loaded_loras = load_loras_from_schedule(self.parsed_prompt, self.loaded_loras)
        return self.loaded_loras


def parse_prompt_schedules(prompt):
    return PromptSchedule(prompt)
