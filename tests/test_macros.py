from textwrap import dedent

import pytest

from prompt_control.macros import expand_macros, expand_segs


@pytest.mark.parametrize(
    "text, result",
    [
        ("DEF(X(a;b)=$1 $2 $3 d)X(A) X(A;B;C)", "A b $3 d A B C d"),
        (
            "DEF(MACRO()=[empty:$1:$2])MACRO MACRO(;) MACRO(;0.5) MACRO(a;0.5)",
            "[empty::$2] [empty::] [empty::0.5] [empty:a:0.5]",
        ),
        ("DEF(X=$1)DEF(Y()=$1)[X Y][X() Y()][X(1) Y(1)]", "[$1 ][ ][1 1]"),
        (
            "DEF(C_ANIMAL=cat)DEF(D_ANIMAL=dog)DEF(IT=It is a $1_ANIMAL $10_ANIMAL)IT(D) IT(C)",
            "It is a dog $10_ANIMAL It is a cat $10_ANIMAL",
        ),
    ],
)
def test_basic_macro(text, result):
    assert expand_macros(text) == result


def test_macro_recursion():
    with pytest.raises(ValueError) as c:
        expand_macros("DEF(X=recurse Y) DEF(Y=recurse X) X")
    assert "Unable to resolve DEFs" in str(c.value)


def test_parsing_cornercase():
    r = expand_macros("This should not get stuck DEF(")
    assert r == "This should not get stuck DEF("


@pytest.mark.parametrize(
    "input, output",
    [
        (
            """\
      A red $b and
      a blue $a
      SEG(a)
      cat
      SEG(b)

      dog
      SEG(c)""",
            "A red dog and\na blue cat",
        ),
        (
            """\
    $a and $b
    SEG(a)
    cat, $b
    SEG(b)
    dog, $c
    SEG(c)
    tiger
    """,
            "cat, dog, tiger and dog, tiger",
        ),
        (
            """\
    $a
    SEG(a)
    a $b
    SEG(b)
    b $a""",
            "a b a b $a",
        ),
    ],
)
def test_segments(input, output):
    assert expand_segs(dedent(input)) == output
