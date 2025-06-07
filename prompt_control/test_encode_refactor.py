import unittest
import numpy.testing as npt

clip_l = None
dual = None


def run(f, *args):
    return getattr(f, f.FUNCTION)(*args)


class TestEncode(unittest.TestCase):
    def tensorsEqual(self, t1, t2):
        npt.assert_equal(t1.detach().numpy(), t2.detach().numpy())

    def condEqual(self, c1, c2, key=None, key_assert=None):
        self.assertEqual(len(c1), len(c2))
        for i in range(len(c1)):
            a, b = c1[i], c2[i]
            if key:
                (key_assert or self.assertEqual)(a[1][key], b[1][key])
            else:
                self.tensorsEqual(a[0], b[0])

    def test_styles(self):
        pc = PCTextEncode()
        for k, clip in [("l", clip_l), ("dual", dual)]:
            for style in ["comfy++", "A1111", "comfy++", "compel", "down_weight"]:
                with self.subTest(f"TE {k} style {style} does not fail when encoding weights"):
                    for normalization in ["none", "mean", "length", "length+mean"]:
                        with self.subTest(f"TE {k} style {style} normalization {normalization}"):
                            (c,) = run(
                                pc,
                                clip,
                                f"STYLE(old+{style}, {normalization}) this prompt has weights, (a:1.2) (b:1.2)",
                            )
                            (c2,) = run(
                                pc,
                                clip,
                                f"STYLE({style}, {normalization}) this prompt has weights, (a:1.2) (b:1.2)",
                            )
                            self.condEqual(c, c2)


if __name__ == "__main__":
    print("Loading ComfyUI")
    import main

    id(main)  # get rid of flake warning
    import nodes
    from .nodes_base import PCTextEncode

    (clip_l,) = nodes.CLIPLoader().load_clip("clip_l.safetensors")
    (dual,) = nodes.DualCLIPLoader().load_clip("clip_l.safetensors", "clip_g.safetensors", "sdxl")
    print("Starting tests")
    unittest.main()
