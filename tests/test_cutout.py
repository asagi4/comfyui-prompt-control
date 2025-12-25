import pytest

from prompt_control.cutoff_parser import parse_cuts


@pytest.fixture(scope="module", autouse=True)
def parser():
    return parse_cuts


def test_parse_no_cuts(parser):
    prompt, cutouts = parse_cuts("a b c")
    assert prompt == "a b c"
    assert cutouts == []


def test_parse_cuts(parser):
    prompt, cutouts = parse_cuts("a [CUT:b:d:0] c")
    assert prompt == "a b c"
    assert cutouts == [("b", "d", 0.0, None, None, None)]


def test_parse_cuts_multiple(parser):
    prompt, cutouts = parse_cuts("a [CUT:b:d:0] [CUT:c:e:1.0:0.5:0.9:-]")
    assert prompt == "a b c"
    assert cutouts == [("b", "d", 0.0, None, None, None), ("c", "e", 1.0, 0.5, 0.9, "-")]
