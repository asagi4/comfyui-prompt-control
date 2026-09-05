from prompt_control import utils


def test_smart_split():
    assert utils.smarter_split(",", "foo,bar") == ["foo", "bar"]
    assert utils.smarter_split(",", "(foo,bar),zonk") == ["(foo,bar)", "zonk"]
    assert utils.smarter_split(",", r"\(foo,bar),zonk") == [r"\(foo", "bar)", "zonk"]
