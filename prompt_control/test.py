import unittest
from .parser import parse_prompt_schedules as parse


class TestParser(unittest.TestCase):
    def assertPrompt(self, p, step, text):
        self.assertEqual(p[0], step)
        self.assertEqual(p[1]["prompt"], text)

    def test_no_scheduling(self):
        p = parse("This is a (basic:0.6) (prompt) with [no scheduling] features")
        expected = [1.0, {"prompt": "This is a (basic:0.6) (prompt) with [no scheduling] features", "loras": {}}]
        self.assertEqual(p.at_step(0), expected)
        self.assertEqual(p.at_step(0.5), expected)
        self.assertEqual(p.at_step(1), expected)

    def test_basic(self):
        p = parse(
            "This is a (basic:0.6) (prompt) with (very [[simple]:(basic:0.6):0.5]:1.1) [features::0.8][ and this is ignored:1]"
        )
        expected = [0.5, {"prompt": "This is a (basic:0.6) (prompt) with (very [simple]:1.1) features", "loras": {}}]
        expected2 = [
            0.8,
            {"prompt": "This is a (basic:0.6) (prompt) with (very (basic:0.6):1.1) features", "loras": {}},
        ]
        expected3 = [1.0, {"prompt": "This is a (basic:0.6) (prompt) with (very (basic:0.6):1.1) ", "loras": {}}]
        self.assertEqual(p.at_step(0), expected)
        self.assertEqual(p.at_step(0.5), expected)
        self.assertEqual(p.at_step(0.7), expected2)
        self.assertEqual(p.at_step(1), expected3)

    def test_lora(self):
        p = parse("This is a (lora:0.6) (prompt) with [no scheduling] features <lora:foo:0.5> <lora:bar:0.5:1.0>")
        expected = [
            1.0,
            {
                "prompt": "This is a (lora:0.6) (prompt) with [no scheduling] features  ",
                "loras": {"foo": {"weight": 0.5, "weight_clip": 0.5}, "bar": {"weight": 0.5, "weight_clip": 1.0}},
            },
        ]
        self.assertEqual(p.at_step(0), expected)
        self.assertEqual(p.at_step(0.5), expected)
        self.assertEqual(p.at_step(1), expected)

    def test_scheduled_lora(self):
        p = parse(
            "This is a (lora:0.6) (prompt) with [scheduling] features [<lora:foo:0.5>:<lora:bar:0.5:0.2>:0.3] <lora:bar:0.5:1.0>"
        )
        expected = [
            0.3,
            {
                "prompt": "This is a (lora:0.6) (prompt) with [scheduling] features  ",
                "loras": {"foo": {"weight": 0.5, "weight_clip": 0.5}, "bar": {"weight": 0.5, "weight_clip": 1.0}},
            },
        ]
        expected2 = [
            1.0,
            {
                "prompt": "This is a (lora:0.6) (prompt) with [scheduling] features  ",
                "loras": {"bar": {"weight": 1.0, "weight_clip": 1.2}},
            },
        ]
        self.assertEqual(p.at_step(0.1), expected)
        self.assertEqual(p.at_step(1), expected2)

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
        for k in prompts:
            self.assertEqual(p.at_step(k), [k, {"prompt": prompts[k], "loras": {}}])

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

        self.assertEqual(
            p.at_step(0.6), [0.7, {"prompt": "This prompt is ", "loras": {"cool": {"weight": 1.0, "weight_clip": 1.0}}}]
        )
        self.assertEqual(
            p.at_step(0.7), [0.7, {"prompt": "This prompt is ", "loras": {"cool": {"weight": 1.0, "weight_clip": 1.0}}}]
        )
        p2 = p.with_filters(filters="hr, xyz")

        self.assertEqual(p2.at_step(0), p2.at_step(1))

    def test_def(self):
        p = parse("DEF(X=0.5) [a:b:X] DEF(test=[c:X]) test test")
        prompts = {
            0.2: (0.5, "a   "),
            0.6: (1.0, "b  c c"),
        }
        for k in prompts:
            self.assertEqual(p.at_step(k), [prompts[k][0], {"prompt": prompts[k][1], "loras": {}}])

    def test_misc(self):
        p = parse("[[a:c:0.5]:0.7]")
        p2 = parse("[:[a:c:0.5]:0.7]")
        self.assertEqual(p.parsed_prompt, p2.parsed_prompt)
        p = parse("test [[a:[b<lora:test:0.5>:0.6]:0.5]:HR]")
        p2 = parse("test [:[a:[:b<lora:test:0.5>:0.6]:0.5]:HR]")
        self.assertEqual(p.parsed_prompt, p2.parsed_prompt)
        pf = p.with_filters(filters="hr")
        self.assertEqual(pf.parsed_prompt, p2.with_filters(filters="hr").parsed_prompt)
        self.assertEqual(pf.at_step(0), [0.5, {"prompt": "test a", "loras": {}}])
        self.assertEqual(pf.at_step(0.55), [0.6, {"prompt": "test ", "loras": {}}])
        self.assertEqual(
            pf.at_step(0.8), [1.0, {"prompt": "test b", "loras": {"test": {"weight": 0.5, "weight_clip": 0.5}}}]
        )
        p = parse("[:[<lora:test:1>:c:0.5]:0.3]")
        self.assertEqual(p.at_step(0), [0.3, {"prompt": "", "loras": {}}])
        self.assertEqual(p.at_step(0.4), [0.5, {"prompt": "", "loras": {"test": {"weight": 1.0, "weight_clip": 1.0}}}])
        self.assertEqual(p.at_step(1.0), [1.0, {"prompt": "c", "loras": {}}])

        p = parse("an [<emb:foo>:<emb:bar>:0.5]")
        prompts = {
            0.2: (0.5, "an embedding:foo"),
            0.8: (1.0, "an embedding:bar"),
        }
        for k in prompts:
            self.assertEqual(p.at_step(k), [prompts[k][0], {"prompt": prompts[k][1], "loras": {}}])

    def test_alternating(self):
        p = parse("[cat|dog|tiger]")
        p2 = parse("[cat|dog|tiger:0.1]")
        p3 = parse("[cat|[dog|wolf]|tiger]")
        p4 = parse("[cat|[dog:wolf<lora:canine:1>:0.5]:0.2]")

        self.assertEqual(p.parsed_prompt, p2.parsed_prompt)
        for i, x in enumerate(["cat", "wolf", "tiger", "cat", "dog", "tiger", "cat", "wolf", "tiger", "cat"]):
            step = round((i * 0.1) + 0.1, 2)
            self.assertPrompt(p3.at_step(step), step, x)

        for i, x in enumerate(["cat", "dog", "cat", "wolf", "cat"]):
            step = round((i * 0.2) + 0.2, 2)
            self.assertPrompt(p4.at_step(step), step, x)
        self.assertEqual(p4.at_step(0.7)[1]["loras"], {"canine": {"weight": 1.0, "weight_clip": 1.0}})


if __name__ == "__main__":
    unittest.main()
