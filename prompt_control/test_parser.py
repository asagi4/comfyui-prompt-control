import unittest
from .parser import parse_prompt_schedules as parse, expand_macros


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

    def test_quote(self):
        p = parse('This is a text with a "QUOTED DEF(X=Y)"')
        expected = prompt(1.0, 'This is a text with a "QUOTED DEF(X=Y)"')
        self.assertEqual(p.at_step(0), expected)
        self.assertEqual(p.at_step(0.5), expected)
        self.assertEqual(p.at_step(1), expected)

    def test_equivalences(self):
        eqs = [
            [parse(p) for p in ["[a:0.1]", "[:a:0.1]", "[:a:0,0.1]", "[:a::0.1,1.0]", "[:a::0.1]"]],
            [parse(p) for p in ["[before:during:after:0.1]", "[before:during:after:0.1,1.0]", "[before:during:0.1]"]],
            [parse(p) for p in ["[a:0.1,0.5]", "[[a:0.1]::0.5]", "[:a::0.1,0.5]", "[a::0.1,0.5]"]],
            [parse(p) for p in ["[a:b:0.5]", "[a::b:0.5,0.5]"]],
            [parse(p) for p in ["[a::0.5]", "[a:::0.5,0.5]"]],
        ]
        for group in eqs:
            for p in group[1:]:
                with self.subTest(p):
                    self.assertEqual(group[0].parsed_prompt, p.parsed_prompt)

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

    def test_range(self):
        p = parse("test [excluded::excluded2:0.1,0.4] test")
        self.assertPrompt(p, 0, 0.1, "test excluded test")
        self.assertPrompt(p, 0.2, 0.4, "test  test")
        self.assertPrompt(p, 0.45, 1.0, "test excluded2 test")
        p = parse("test [[:included::0.2,0.8]|[excluded::excluded2:0.4,0.9]:0.1] test")
        self.assertPrompt(p, 0, 0.1, "test  test")
        self.assertPrompt(p, 0.25, 0.3, "test included test")
        self.assertPrompt(p, 0.15, 0.2, "test excluded test")
        self.assertPrompt(p, 0.25, 0.3, "test included test")
        self.assertPrompt(p, 0.55, 0.6, "test  test")
        self.assertPrompt(p, 0.95, 1.0, "test excluded2 test")

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

        p = parse("DEF(X=[($1):($1:$2):$2])X(test;0.7)")
        p2 = parse("[(test):(test:0.7):0.7]")
        with self.subTest("parameters"):
            self.assertEqual(p.parsed_prompt, p2.parsed_prompt)

        p = parse("DEF(X=[($1):($1:$2):$2])DEF(Y=X(test;$1))Y(0.7) Y(0.5)")
        p2 = parse("[(test):(test:0.7):0.7] [(test):(test:0.5):0.5]")
        with self.subTest("two functions"):
            self.assertEqual(p.parsed_prompt, p2.parsed_prompt)

        p = expand_macros("DEF(X(a;b)=$1 $2 $3 d)X(A) X(A;B;C)")
        with self.subTest("defaults"):
            self.assertEqual(p, "A b $3 d A B C d")

        p = expand_macros("DEF(MACRO()=[empty:$1:$2])MACRO MACRO(;) MACRO(;0.5) MACRO(a;0.5)")
        with self.subTest("Empty default for $1"):
            self.assertEqual(p, "[empty::$2] [empty::] [empty::0.5] [empty:a:0.5]")

        p = expand_macros("DEF(X=$1)DEF(Y()=$1)[X Y][X() Y()][X(1) Y(1)]")
        with self.subTest("defaults, DEF=X vs DEF=X()"):
            self.assertEqual(p, "[$1 ][ ][1 1]")

        p = parse("DEF(test(1)=prompt $1)DEF(test2((a); (test))=[$1:$2:0.5])test test2")
        p2 = parse("prompt 1 [(a):(prompt 1):0.5]")
        with self.subTest("defaults, nested parens"):
            self.assertEqual(p.parsed_prompt, p2.parsed_prompt)

        with self.assertRaises(ValueError) as c:
            expand_macros("DEF(X=recurse Y) DEF(Y=recurse X) X")
        self.assertTrue("Unable to resolve DEFs" in str(c.exception))

    def test_escapes(self):
        p = parse(r"[a:\:a:0.5] :\[a:b:0.5]")
        self.assertPrompt(p, 0, 0.5, r"a :\[a:b:0.5]")
        self.assertPrompt(p, 0.55, 1, r":a :\[a:b:0.5]")

        p = parse(r"[embedding\:a:embedding\:b:0.1,0.5]")
        self.assertPrompt(p, 0.15, 0.5, r"embedding:a")
        self.assertPrompt(p, 0.55, 1, r"embedding:b")

        p = parse(r"[embedding\:a:embedding\:b:embedding\:c:0.1,0.5]")
        self.assertPrompt(p, 0.0, 0.1, r"embedding:a")
        self.assertPrompt(p, 0.15, 0.5, r"embedding:b")
        self.assertPrompt(p, 0.55, 1, r"embedding:c")

        p = parse(r"[a\:b\\:c:0.5]")
        self.assertPrompt(p, 0.0, 0.5, "a:b\\")
        self.assertPrompt(p, 0.55, 1, r"c")

        p = parse(r"[a:\#b:0.5]")
        self.assertPrompt(p, 0.0, 0.5, "a")
        self.assertPrompt(p, 0.55, 1, "#b")

    def test_comments(self):
        p = parse("this is a # comment")
        self.assertPrompt(p, 0, 1.0, "this is a ")
        p = parse("this is a [comment#:scheduled:0.6]")
        self.assertPrompt(p, 0, 1.0, "this is a [comment")
        p = parse(r"this is a [comment\#:scheduled:0.6]")
        self.assertPrompt(p, 0, 0.6, "this is a comment#")
        self.assertPrompt(p, 0.65, 1.0, "this is a scheduled")
        p = parse("#this is a comment\nthis is a prompt")
        self.assertPrompt(p, 0, 1.0, "\nthis is a prompt")

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
            with self.subTest(step):
                self.assertPrompt(p3, step, step, x)

        for i, x in enumerate([["cat"], ["dog"], ["cat"], ["wolf", ("canine", 1.0, 1.0)], ["cat"]]):
            step = round((i * 0.2) + 0.2, 2)
            with self.subTest(step):
                self.assertPrompt(p4, step, step, *x)
        self.assertPrompt(p4, 0.7, 0.8, "wolf", ("canine", 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
