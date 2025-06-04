import unittest

clip_l = None
dual = None


def run(f, *args):
    return getattr(f, f.FUNCTION)(*args)


class TestEncode(unittest.TestCase):
    def condEqual(self, c1, c2):
        self.assertEqual(len(c1), len(c2))
        for i in range(len(c1)):
            self.assertTrue((c1[i][0] == c2[i][0]).all())

    def test_basic_encode(self):
        pc = PCTextEncode()
        comfy = nodes.CLIPTextEncode()
        concat = nodes.ConditioningConcat()
        for k, clip in [("l", clip_l), ("dual", dual)]:
            with self.subTest(k):
                (c1,) = run(pc, clip, "test")
                (c2,) = run(comfy, clip, "test")
                self.condEqual(c1, c2)

                (c3,) = run(pc, clip, "test CAT test")
                (c4,) = run(concat, c2, c2)
                self.condEqual(c3, c4)


if __name__ == "__main__":
    print("Loading ComfyUI")
    import main

    id(main)  # get rid of flake warning
    import nodes
    from .nodes_base import PCTextEncode

    (clip_l,) = nodes.CLIPLoader().load_clip("clip_l.safetensors")
    (dual,) = nodes.DualCLIPLoader().load_clip("clip_l.safetensors", "t5xxl_fp16.safetensors", "flux")
    print("Starting tests")
    unittest.main()
