import unittest
from .parser import parse_prompt_schedules as parse


def prompt(until, text, *loras):
    loras = {lora: {"weight": unet, "weight_clip": te} for lora, unet, te in loras}
    return [until, {"prompt": text, "loras": loras}]


class TestParser(unittest.TestCase):
    def assertPrompt(self, p, at, until, text, *loras):
        self.assertEqual(p.at_step(at), prompt(until, text, *loras))

    def test_no_scheduling(self):
        p = parse("This is a (basic:0.6) (prompt) with [no scheduling] features")
        expected = prompt(1.0, "This is a (basic:0.6) (prompt) with [no scheduling] features")
        self.assertEqual(p.at_step(0), expected)
        self.assertEqual(p.at_step(0.5), expected)
        self.assertEqual(p.at_step(1), expected)

    def test_basic(self):
        p = parse(
            "This is a (basic:0.6) (prompt) with (very [[simple]:(basic:0.6):0.5]:1.1) [features::0.8][ and this is ignored:1]"
        )
        self.assertPrompt(p, 0, 0.5, "This is a (basic:0.6) (prompt) with (very [simple]:1.1) features")
        self.assertPrompt(p, 0.5, 0.5, "This is a (basic:0.6) (prompt) with (very [simple]:1.1) features")
        self.assertPrompt(p, 0.7, 0.8, "This is a (basic:0.6) (prompt) with (very (basic:0.6):1.1) features")
        self.assertPrompt(p, 1.0, 1.0, "This is a (basic:0.6) (prompt) with (very (basic:0.6):1.1) ")

    def test_lora(self):
        p = parse("This is a (lora:0.6) (prompt) with [no scheduling] features <lora:foo:0.5> <lora:bar:0.5:1.0>")
        expected = prompt(
            1.0, "This is a (lora:0.6) (prompt) with [no scheduling] features  ", ("foo", 0.5, 0.5), ("bar", 0.5, 1.0)
        )
        self.assertEqual(p.at_step(0), expected)
        self.assertEqual(p.at_step(0.5), expected)
        self.assertEqual(p.at_step(1), expected)

    def test_scheduled_lora(self):
        p = parse(
            "This is a (lora:0.6) (prompt) with [scheduling] features [<lora:foo:0.5>:<lora:bar:0.5:0.2>:0.3] <lora:bar:0.5:1.0>"
        )
        self.assertPrompt(
            p,
            0.1,
            0.3,
            "This is a (lora:0.6) (prompt) with [scheduling] features  ",
            ("foo", 0.5, 0.5),
            ("bar", 0.5, 1.0),
        )
        self.assertPrompt(p, 0.5, 1.0, "This is a (lora:0.6) (prompt) with [scheduling] features  ", ("bar", 1.0, 1.2))

    def test_seq(self):
        p = parse("This is a sequence of [SEQ:a:0.2::0.5:c:0.8][SEQ: and x:0.8]")
        p2 = parse("This is a sequence of [[a:[c:0.5]:0.2]::0.8][ and x::0.8]")
        prompts = {
            0.2: "This is a sequence of a and x",
            0.5: "This is a sequence of  and x",
            0.8: "This is a sequence of c and x",
            1.0: "This is a sequence of ",
        }
        self.assertEqual(p.parsed_prompt, p2.parsed_prompt)
        for k, v in prompts.items():
            self.assertPrompt(p, k, k, v)

    def test_shortcuts_scheduling(self):
        p = parse("A schedule [a:0.1,0.7] b")
        p2 = parse("A schedule [[a:0.1]::0.7] b")
        p3 = parse("A schedule [a:b:0.5,0.8]")
        p4 = parse("A schedule [[a:0.5]:b:0.8]")
        self.assertEqual(p.parsed_prompt, p2.parsed_prompt)
        self.assertEqual(p3.parsed_prompt, p4.parsed_prompt)

    def test_nested(self):
        p = parse(
            "This [prompt is [SEQ:[crazy:weird:0.2] stuff:0.5:<lora:cool:1>:0.7:nesting:1.0]:completely ignored with tags:HR]"
        )
        prompts = {
            0.2: (0.2, "This prompt is crazy stuff"),
            0.3: (0.5, "This prompt is weird stuff"),
            0.5: (0.5, "This prompt is weird stuff"),
            0.8: (1.0, "This prompt is nesting"),
        }
        for k in prompts:
            self.assertEqual(p.at_step(k), [prompts[k][0], {"prompt": prompts[k][1], "loras": {}}])

        self.assertPrompt(p, 0.6, 0.7, "This prompt is ", ("cool", 1.0, 1.0))
        self.assertPrompt(p, 0.7, 0.7, "This prompt is ", ("cool", 1.0, 1.0))
        p2 = p.with_filters(filters="hr, xyz")

        self.assertEqual(p2.at_step(0), p2.at_step(1))

    def test_def(self):
        p = parse("DEF(X=0.5) [a:b:X] DEF(test = [c:X]) test test")
        prompts = {
            0.2: (0.5, "a   "),
            0.6: (1.0, "b  c c"),
        }
        for k, v in prompts.items():
            self.assertPrompt(p, k, v[0], v[1])

    def test_misc(self):
        p = parse("[[a:c:0.5]:0.7]")
        p2 = parse("[:[a:c:0.5]:0.7]")
        self.assertEqual(p.parsed_prompt, p2.parsed_prompt)

        p = parse("test [[a:[b<lora:test:0.5>:0.6]:0.5]:HR]")
        p2 = parse("test [:[a:[:b<lora:test:0.5>:0.6]:0.5]:HR]")
        self.assertEqual(p.parsed_prompt, p2.parsed_prompt)

        pf = p.with_filters(filters="hr")
        self.assertEqual(pf.parsed_prompt, p2.with_filters(filters="hr").parsed_prompt)
        self.assertPrompt(pf, 0, 0.5, "test a")
        self.assertPrompt(pf, 0.55, 0.6, "test ")
        self.assertPrompt(pf, 0.8, 1.0, "test b", ("test", 0.5, 0.5))

        p = parse("[:[<lora:test:1>:c:0.5]:0.3]")
        self.assertPrompt(p, 0, 0.3, "")
        self.assertPrompt(p, 0.4, 0.5, "", ("test", 1.0, 1.0))
        self.assertPrompt(p, 1.0, 1.0, "c")

        p = parse("an [<emb:foo>:<emb:bar>:0.5]")
        prompts = {
            0.2: (0.5, "an embedding:foo"),
            0.8: (1.0, "an embedding:bar"),
        }
        for k, v in prompts.items():
            self.assertPrompt(p, k, v[0], v[1])

    def test_alternating(self):
        p = parse("[cat|dog|tiger]")
        p2 = parse("[cat|dog|tiger:0.1]")
        p3 = parse("[cat|[dog|wolf]|tiger]")
        p4 = parse("[cat|[dog:wolf<lora:canine:1>:0.5]:0.2]")

        self.assertEqual(p.parsed_prompt, p2.parsed_prompt)
        for i, x in enumerate(["cat", "wolf", "tiger", "cat", "dog", "tiger", "cat", "wolf", "tiger", "cat"]):
            step = round((i * 0.1) + 0.1, 2)
            self.assertPrompt(p3, step, step, x)

        for i, x in enumerate([["cat"], ["dog"], ["cat"], ["wolf", ("canine", 1.0, 1.0)], ["cat"]]):
            step = round((i * 0.2) + 0.2, 2)
            self.assertPrompt(p4, step, step, *x)
        self.assertPrompt(p4, 0.7, 0.8, "wolf", ("canine", 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
