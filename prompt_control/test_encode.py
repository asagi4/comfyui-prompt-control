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

    def test_basic_encode(self):
        pc = PCTextEncode()
        comfy = nodes.CLIPTextEncode()
        combine = nodes.ConditioningCombine()
        concat = nodes.ConditioningConcat()
        zeroout = nodes.ConditioningZeroOut()
        for k, clip in [("l", clip_l), ("dual", dual)]:
            with self.subTest(k):
                with self.subTest("No exceptions"):
                    run(
                        pc,
                        clip,
                        "test AND test (test:1.2) BREAK test AND TE_WEIGHT(all=0) SDXL() AND AREA(,,) test CAT test",
                    )
                with self.subTest("Basic"):
                    (c1,) = run(pc, clip, "test")
                    (c2,) = run(comfy, clip, "test")
                    c = c2  # Used in later tests
                    self.condEqual(c1, c2)

                    (c1,) = run(pc, clip, "(test:1.2)")
                    (c2,) = run(comfy, clip, "(test:1.2)")

                with self.subTest("Concat"):
                    (c1,) = run(pc, clip, "test CAT test")
                    (c2,) = run(concat, c, c)
                    self.condEqual(c1, c2)

                with self.subTest("Combine"):
                    (c1,) = run(pc, clip, "test AND test")
                    (c2,) = run(combine, c, c)
                    self.condEqual(c1, c2)

                with self.subTest("Zero out"):
                    (c1,) = run(pc, clip, "test TE_WEIGHT(all=0)")
                    (c2,) = run(zeroout, c)
                    self.condEqual(c1, c2)

    def test_styles(self):
        pc = PCTextEncode()
        comfy = nodes.CLIPTextEncode()
        for k, clip in [("l", clip_l), ("dual", dual)]:
            (no_weights,) = run(comfy, clip, "this prompt has no weights")
            for style in ["comfy", "A1111", "comfy++", "compel", "down_weight", "perp"]:
                with self.subTest(f"TE {k} style {style} no weights equal comfy"):
                    (c,) = run(pc, clip, "this prompt has no weights")
                    self.condEqual(no_weights, c)
                with self.subTest(f"TE {k} style {style} does not fail when encoding weights"):
                    for normalization in ["none", "mean", "length", "mean+length", "length+mean"]:
                        with self.subTest(f"TE {k} style {style} normalization {normalization}"):
                            (c,) = run(
                                pc,
                                clip,
                                f"STYLE({style}, {normalization}) (this prompt) (has weights:0.9), (a:1.2) (b:1.2)",
                            )

    def test_masks(self):
        pc = PCTextEncode()
        comfy = nodes.CLIPTextEncode()
        solidmask = comfy_extras.nodes_mask.SolidMask()
        setMask = nodes.ConditioningSetMask()
        for k, clip in [("l", clip_l), ("dual", dual)]:
            (c1,) = run(pc, clip, "test MASK()")
            (c2,) = run(comfy, clip, "test")
            (c2,) = run(setMask, c2, run(solidmask, 1.0, 512, 512)[0], "default", 1.0)
            self.condEqual(c1, c2)
            self.condEqual(c1, c2, "mask", self.tensorsEqual)


if __name__ == "__main__":
    print("Loading ComfyUI")
    import main

    id(main)  # get rid of flake warning
    import nodes
    import comfy_extras.nodes_mask
    from .nodes_base import PCTextEncode

    (clip_l,) = nodes.CLIPLoader().load_clip("clip_l.safetensors")
    (dual,) = nodes.DualCLIPLoader().load_clip("clip_l.safetensors", "t5xxl_fp16.safetensors", "flux")
    print("Starting tests")
    unittest.main()
