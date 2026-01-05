from __future__ import annotations

import itertools as it
from dataclasses import dataclass
from math import ceil
from typing import Any, TypeAlias

from typing_extensions import override

from .macros import expand_macros
from .parsy import any_char, char_from, digit, eof, forward_declaration, generate, regex, seq, string, success

FOREVER = float("inf")

EvalResult: TypeAlias = tuple[float, str, list["LoRA"]]


def merge_until(i: EvalResult, minimum: float):
    until, p, loras = i
    until = min(until, minimum)
    return until, p, loras


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 2) → AB CD EF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(it.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


EvalResult: TypeAlias = tuple[float, str, list["LoRA"]]


class Expression:
    def eval(self, step: float, tags: list[str]) -> EvalResult:
        return (FOREVER, "", [])

    def required_steps(self, max_steps: float) -> set[float]:
        return set()


@dataclass
class Text(Expression):
    string: str

    @override
    def eval(self, step: float, tags: list[str]) -> EvalResult:
        assert isinstance(self.string, str)
        return FOREVER, self.string, []


@dataclass
class Alternate(Expression):
    prompts: list[Expression]
    step: float = 0.1

    @override
    def eval(self, step: float, tags: list[str]) -> EvalResult:
        SCALE = 10_000
        step = max(step, self.step)
        position = (step * SCALE) / (self.step * SCALE)
        idx = (ceil(position) - 1) % len(self.prompts)

        r = self.prompts[max(0, idx)].eval(step, tags)
        r = merge_until(r, max(self.step, ceil(position) * self.step))
        return r

    @override
    def required_steps(self, max_steps: float):
        r = set()
        for x in self.prompts:
            r.update(x.required_steps(max_steps))
        r.update(set(x / 100 for x in range(0, int(max_steps * 100), int(self.step * 100))))
        return r


@dataclass
class Sequence(Expression):
    prompts: list[tuple[Expression, float]]

    @override
    def eval(self, step: float, tags: list[str]) -> EvalResult:
        item = Text("")
        found_step = FOREVER
        for prompt, switch_step in self.prompts:
            if step <= switch_step:
                found_step = switch_step
                item = prompt
                break

        return merge_until(item.eval(step, tags), found_step)

    @override
    def required_steps(self, max_steps: float):
        return set(step for _, step in self.prompts if step <= max_steps)


@dataclass
class Schedule(Expression):
    before: Prompt
    during: Prompt
    after: Prompt
    start: float
    end: float
    tag: str | None

    def tag_matches(self, tags: list[str]):
        return self.tag in tags

    @override
    def eval(self, step: float, tags: list[str]) -> EvalResult:
        if self.tag is not None and not self.tag_matches(tags):
            return self.before.eval(step, tags)
        if self.tag_matches(tags):
            return self.during.eval(step, tags)

        if step <= self.start:
            return merge_until(self.before.eval(step, tags), self.start)
        if self.start < step <= self.end:
            return merge_until(self.during.eval(step, tags), self.end)
        if step > self.end:
            return self.after.eval(step, tags)
        raise AssertionError("How are you here?")

    @override
    def required_steps(self, max_steps: float):
        r = set()
        if self.tag is not None:
            return r
        if self.start < max_steps:
            r.add(self.start)
        if self.end < max_steps:
            r.add(self.end)
        r.update(self.before.required_steps(max_steps))
        r.update(self.during.required_steps(max_steps))
        r.update(self.after.required_steps(max_steps))
        return r


@dataclass
class Prompt(Expression):
    data: list[Expression]

    @override
    def eval(self, step: float, tags: list[str]) -> EvalResult:
        evals = [x.eval(step, tags) for x in self.data]
        text = "".join(x[1] for x in evals)
        untils = [x[0] for x in evals]
        loras = []
        for x in evals:
            loras.extend(x[2])
        until = FOREVER if not untils else min(untils)
        return until, text, loras

    @override
    def required_steps(self, max_steps):
        r = set()
        for x in self.data:
            r.update(x.required_steps(max_steps))
        return r


@dataclass
class LoRA(Expression):
    filename: str
    w_model: float = 1.0
    w_te: float = 1.0

    def eval(self, step: float, tags: list[str]) -> EvalResult:
        return FOREVER, "", [self]


def find_weight_at(weights: list[tuple[float, float]], step: float, until: float):
    res_w = 0
    for this, next in zip(weights, it.chain(weights[1:], [(0, FOREVER)]), strict=False):
        w, start = this
        _, next_start = next
        if start > step or next_start < step:
            until = min(until, start)
            continue
        res_w = w
    return until, res_w


@dataclass
class LoRACTL(Expression):
    filename: str
    w_model: list[tuple[float, float]]
    w_te: list[tuple[float, float]]

    def eval(self, step: float, tags: list[str]) -> EvalResult:
        until, w1 = find_weight_at(self.w_model, step, FOREVER)
        until, w2 = find_weight_at(self.w_te, step, until)
        lora = []
        if w1 != 0 or w1 != 0:
            lora = [LoRA(self.filename, w1, w2)]

        return until, "", lora

    def required_steps(self, max_steps):
        r = set(x[1] for x in self.w_model)
        r.update(set(x[1] for x in self.w_te))
        return r


def combine_arglist(prompts, start_end) -> Schedule:
    a, b, c = prompts
    start_or_tag, end = start_end
    empty = Prompt([])
    start = start_or_tag
    # Handle [a:b:TAG]
    if isinstance(start_or_tag, str):
        if b is None:
            before = empty
            during = a  # [a:TAG] produces a when tag is active
        else:
            before, during = a, b  # [a:b:TAG] changes from a to b when tag is active
        return Schedule(before, during, empty, start=0.0, end=FOREVER, tag=start_or_tag)
    during = before = after = empty
    if end is not None:
        if b is None:  # [a:0,0.5] == [:a:0,0.5]
            during = a
            before = after = empty
        elif c is None:  # [a:b:0,0.5]
            before = empty
            during = a
            after = b
        else:
            before, during, after = a, b, c
    else:
        end = FOREVER
        if b is None:  # [a:0.5] == [::a:0.5,0.5]
            before = empty
            during = a
            after = a
        else:
            before = a
            during = b
            after = b
            # c always gets ignored
    start = float(start)  # for typechecking
    return Schedule(before, during, after, start, end, tag=None)


def token(s: str):
    return string(s).map(Text)


def combine_prompt(*prompts):
    p = prompts
    if len(p) == 1:
        p = p[0]
    if isinstance(p, Prompt):
        p = p.data[0] if len(p.data) == 1 else combine_prompt(*p.data)
    if isinstance(p, Expression):
        return p
    p = [combine_prompt(x) for x in p]
    return Prompt(p)


@dataclass
class PromptSchedule:
    parse_tree: Expression
    filters: list[str]
    start: float
    end: float
    num_steps: int

    def at_step(self, step: float) -> tuple[float, dict[str, Any]]:
        max_step = self.num_steps or 1.0
        if max_step > 1 and step < 1:
            step = step * max_step
        until, p, lora_list = self.parse_tree.eval(step, self.filters)
        loras = {}
        for lora in lora_list:
            d = loras.get(lora.filename, {})
            d["weight"] = d.get("weight", 0) + lora.w_model
            d["weight_clip"] = d.get("weight_clip", 0) + lora.w_te
            loras[lora.filename] = d
        if max_step > 0 and until > 1:
            # TODO: better logic for this?
            until = min(until / max_step, 1.0)
        return (min(max_step, round(until, 2)), {"prompt": p, "loras": loras})

    def with_filters(self, filters: str | None = None, start: float | None = None, end: float | None = None):
        return PromptSchedule(
            self.parse_tree,
            self.filters if filters is None else parse_filters(filters),
            self.start if start is None else start,
            self.end if end is None else end,
            self.num_steps,
        )

    def clone(self):
        return self.with_filters()

    def __iter__(self):
        return (x for x in self.parsed_prompt if x[0] != 0)

    @property
    def parsed_prompt(self):
        max_step = self.num_steps or 1.0
        required_steps = self.parse_tree.required_steps(max_step).union({max_step})

        prompts = list(sorted((self.at_step(step) for step in required_steps), key=lambda x: x[0]))
        res = []
        prev_end = -1
        for end_at, p in prompts:
            if end_at < self.start:
                continue
            elif end_at < self.end and prev_end < end_at:
                res.append([end_at, p])
                prev_end = end_at
            elif end_at >= self.end and prev_end < self.end:
                res.append([end_at, p])
                break

        if len(res) == 0:
            res = [[1.0], prompts[-1][1]]

        return res


def lora_weights(p):
    @generate
    def parser():
        w_model = yield col >> p
        w_te = yield (col >> p).optional(w_model)
        return [w_model, w_te]

    return parser.desc("lora_weights")


prompt = forward_declaration()
empty = Text("")
comma = token(",")
col = token(":")
lsq = token("[")
rsq = token("]")
lpar = token("(")
rpar = token(")")
tag = regex(r"[A-Z_]+")
non_special = regex(r"[^:\[\]()|\\<>#]+").map(Text)
filename = regex(r"[^:<>]+")

comment = string("#") >> any_char.until(eof | char_from("\n")) >> success(empty)
escape = (string("\\") >> char_from("\\[]:#")).map(Text)
emphasis = seq(lpar, (prompt | col).at_least(0), rpar)
number = (digit.at_least(1) + string(".") * 1 + digit.many() | digit.at_least(1)).concat().map(float)

opt_prompt = prompt.optional(empty)
step_range = seq(number | tag, (comma >> number).optional())
arglist = seq((opt_prompt << col).optional() * 3, step_range)
schedule = lsq >> arglist.combine(combine_arglist) << rsq

alternate = (lsq >> seq(prompt.sep_by(string("|"), min=1), (col >> number).optional(0.1)) << rsq).combine(Alternate)
sequence = (lsq >> string("SEQ") >> seq(col >> opt_prompt << col, number).at_least(1) << rsq).map(Sequence)
bracketed = seq(lsq, prompt.at_least(0), rsq) | sequence | schedule | alternate
lora = (string("<lora:") >> filename * 1 + lora_weights(number) << string(">")).combine(LoRA)
# TODO: fix
ctlweight = seq(number, (string("@") >> number).optional(0)).sep_by(comma, min=1)
loractl = (string("<loractl:") >> filename * 1 + lora_weights(ctlweight) << string(">")).combine(LoRACTL)
emb = (string("<emb:") >> filename << string(">")).map(lambda f: Text(f"embedding:{f}"))

expr = escape | comment | non_special | bracketed | emphasis.combine(combine_prompt) | lora | loractl | emb
prompt_ = expr.at_least(1).combine(combine_prompt)
prompt.become(prompt_)
# Treat any character that isn't valid prompt syntax as just text
all = (prompt | any_char.map(Text)).at_least(0).combine(combine_prompt)


def parse_filters(filters: str):
    return [x.strip().upper() for x in filters.split(",") if x.strip()]


def parse(text):
    return combine_prompt(all.parse(text))


def parse_prompt_schedules(text, filters="", start=0, end=1.0, num_steps=0):
    return PromptSchedule(parse(expand_macros(text.strip())), parse_filters(filters), start, end, num_steps)
