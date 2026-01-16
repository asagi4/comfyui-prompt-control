import unittest
import unittest.mock as mock
import numpy.testing as npt
from os import environ
import nodes
import comfy_extras.nodes_mask
from .nodes_base import PCTextEncode

clips = []

import logging

logging.basicConfig()


def run(f, *args):
    if hasattr(f, "execute"):
        return f.execute(*args)
    else:
        return getattr(f, f.FUNCTION)(*args)


def compare_hookgroup_mask(h1, h2):
    assert len(h1.hooks) == len(h2.hooks)
    for a, b in zip(h1.hooks, h2.hooks):
        assert (a.mask == b.mask).all()


@mock.patch("torch.cuda.current_device", lambda: "cpu")
class TestEncode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global clips
        print("Loading ComfyUI")
        from comfy.sd import load_clip
        from pathlib import Path

        to_test = environ.get("TEST_TE", "clip_l").split()
        model_dir = environ.get("COMFYUI_TE_DIR", ".")

        te_root = Path(model_dir).resolve()

        if "clip_l" in to_test:
            clip_l = load_clip(
                ckpt_paths=[str(te_root / "clip_l.safetensors")], clip_type="stable_diffusion", model_options={}
            )
            clips.append(("clip_l", clip_l))

        if "t5" in to_test:
            dual = load_clip(
                [str(te_root / "clip_l.safetensors"), str(te_root / "t5xxl_fp16.safetensors")],
                clip_type="flux",
                model_options={},
            )
            clips.append(("clip_l+t5", dual))

        print("Starting tests")

    def tensorsEqual(self, t1, t2):
        npt.assert_equal(t1.detach().numpy(), t2.detach().numpy())

    def condEqual(self, c1, c2, key=None, key_assert=None):
        self.assertEqual(len(c1), len(c2))
        for i in range(len(c1)):
            a, b = c1[i], c2[i]
            if key:
                (key_assert or self.assertEqual)(a[1].get(key), b[1].get(key))
            else:
                self.tensorsEqual(a[0], b[0])

    def test_basic_encode(self):
        pc = PCTextEncode()
        comfy = nodes.CLIPTextEncode()
        combine = nodes.ConditioningCombine()
        average = nodes.ConditioningAverage()
        concat = nodes.ConditioningConcat()
        zeroout = nodes.ConditioningZeroOut()
        for k, clip in clips:
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

                with self.subTest("Quotes"):
                    (c1,) = run(pc, clip, 'Text saying "DOG MASK AND CAT COUPLE MASK(X)"')
                    (c2,) = run(comfy, clip, 'Text saying "DOG MASK AND CAT COUPLE MASK(X)"')
                    self.condEqual(c1, c2)

                with self.subTest("Function cornercase"):
                    (c1,) = run(pc, clip, "test SDXL function")
                    (c2,) = run(comfy, clip, "test SDXL function")
                    (c3,) = run(pc, clip, "test SDXL() function")
                    self.condEqual(c1, c2)

                with self.subTest("Weights"):
                    (c1,) = run(pc, clip, "(test:1.2) (test:0.6)")
                    (c2,) = run(comfy, clip, "(test:1.2) (test:0.6)")
                    self.condEqual(c1, c2)

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

                with self.subTest("Average"):
                    (c1,) = run(comfy, clip, "test1")
                    (c2,) = run(comfy, clip, "test2")
                    (c3,) = run(pc, clip, "test1 AVG() test2")
                    (c4,) = run(pc, clip, "test1 AVG test2")
                    (avg,) = run(average, c1, c2, 0.5)
                    self.condEqual(avg, c3)
                    self.condEqual(avg, c4)

                with self.subTest("Average multi"):
                    (c1,) = run(comfy, clip, "test1")
                    (c2,) = run(comfy, clip, "test2")
                    (c3,) = run(comfy, clip, "test3")
                    (c4,) = run(pc, clip, "test1 AVG() test2 AVG() test3")
                    (c5,) = run(pc, clip, "test1 AVG test2 AVG test3")
                    (avg1,) = run(average, c1, c2, 0.5)
                    (avg,) = run(average, avg1, c3, 0.5)
                    self.condEqual(avg, c4)
                    self.condEqual(avg, c5)

    @unittest.expectedFailure
    def test_failure(self):
        pc = PCTextEncode()
        comfy = nodes.CLIPTextEncode()
        for k, clip in clips:
            with self.subTest(k):
                (c1,) = run(comfy, clip, "test SDXL function")
                (c2,) = run(pc, clip, "test SDXL() function")
                self.condEqual(c1, c2)

    def test_weight(self):
        pc = PCTextEncode()
        comfy = nodes.CLIPTextEncode()
        combine = nodes.ConditioningCombine()
        strength = nodes.ConditioningSetAreaStrength()
        for k, clip in clips:
            (c,) = run(comfy, clip, "test")
            (c2,) = run(strength, c, 0.5)
            with self.subTest(f"Testing {k}"):
                with self.subTest("Conditioning weights"):
                    (a,) = run(pc, clip, "test :0.5 AND test :0.5")
                    (b,) = run(combine, c2, c2)
                    self.condEqual(a, b)
                    self.condEqual(a, b, "strength")
                with self.subTest("Weight == 0"):
                    (a,) = run(pc, clip, "test :0.5 AND test :0 AND test")
                    (b,) = run(combine, c2, c)
                    self.condEqual(a, b)
                    self.condEqual(a, b, "strength")

    def test_attn_couple(self):
        pc = PCTextEncode()
        for k, clip in clips:
            with self.subTest(f"Testing {k}"):
                (c,) = run(pc, clip, "test COUPLE prompt1 AND test2 COUPLE prompt2")
                (c2,) = run(pc, clip, "test COUPLE prompt1 COUPLE test2 COUPLE prompt2")
                self.assertTrue(len(c) == 2)
                self.assertTrue(len(c2) == 1)
            with self.subTest(f"Testing {k} mask shortcut"):
                (c,) = run(pc, clip, "test COUPLE() prompt1")
                (c2,) = run(pc, clip, "test COUPLE MASK() prompt1")
                self.condEqual(c, c2)
                self.condEqual(c, c2, "hooks", compare_hookgroup_mask)
            with self.subTest(f"Testing {k} mask shortcut 2"):
                (c,) = run(pc, clip, "test COUPLE(0 0.2, 0.5) prompt1")
                (c2,) = run(pc, clip, "test COUPLE MASK(0 0.2, 0.5) prompt1")
                self.condEqual(c, c2)
                self.condEqual(c, c2, "hooks", compare_hookgroup_mask)

    def test_styles(self):
        pc = PCTextEncode()
        comfy = nodes.CLIPTextEncode()
        for k, clip in clips:
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
        for k, clip in clips:
            (c1,) = run(pc, clip, "test MASK()")
            (c2,) = run(comfy, clip, "test")
            (c2,) = run(setMask, c2, run(solidmask, 1.0, 512, 512)[0], "default", 1.0)
            self.condEqual(c1, c2)
            self.condEqual(c1, c2, "mask", self.tensorsEqual)


if __name__ == "__main__":
    unittest.main()
