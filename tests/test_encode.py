import numpy.testing as npt
import pytest


def run(f, *args):
    if hasattr(f, "execute"):
        return f.execute(*args)
    else:
        return getattr(f, f.FUNCTION)(*args)


@pytest.fixture(scope="module")
def text_encoder_clips():
    import os
    from pathlib import Path

    from comfy.sd import load_clip

    clips = []
    to_test = os.environ.get("TEST_TE", "clip_l").split()
    model_dir = os.environ.get("COMFYUI_TE_DIR", ".")
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

    return clips


@pytest.fixture
def pc_text_encode():
    from prompt_control.nodes_base import PCTextEncode

    return PCTextEncode()


@pytest.fixture
def node_class_objs():
    import comfy_extras.nodes_mask
    import nodes

    # Return all used node class objects
    return {
        "comfy": nodes.CLIPTextEncode(),
        "combine": nodes.ConditioningCombine(),
        "average": nodes.ConditioningAverage(),
        "concat": nodes.ConditioningConcat(),
        "zeroout": nodes.ConditioningZeroOut(),
        "strength": nodes.ConditioningSetAreaStrength(),
        "solidmask": comfy_extras.nodes_mask.SolidMask(),
        "setmask": nodes.ConditioningSetMask(),
    }


def tensors_equal(t1, t2):
    npt.assert_equal(t1.detach().numpy(), t2.detach().numpy())


def cond_equal(c1, c2, key=None, key_assert=None):
    assert len(c1) == len(c2)
    for i in range(len(c1)):
        a, b = c1[i], c2[i]
        if key:
            (key_assert or assert_equal)(a[1].get(key), b[1].get(key))
        else:
            tensors_equal(a[0], b[0])


def assert_equal(a, b):
    assert a == b


@pytest.mark.usefixtures("text_encoder_clips", "pc_text_encode", "node_class_objs")
class TestPCTextEncode:
    def test_basic_encode(self, text_encoder_clips, pc_text_encode, node_class_objs):
        comfy = node_class_objs["comfy"]
        combine = node_class_objs["combine"]
        average = node_class_objs["average"]
        concat = node_class_objs["concat"]
        zeroout = node_class_objs["zeroout"]

        for _k, clip in text_encoder_clips:
            # No exceptions
            run(
                pc_text_encode,
                clip,
                "test AND test (test:1.2) BREAK test AND TE_WEIGHT(all=0) SDXL() AND AREA(,,) test CAT test",
            )

            # Basic
            (c1,) = run(pc_text_encode, clip, "test")
            (c2,) = run(comfy, clip, "test")
            c = c2  # Used in later tests
            cond_equal(c1, c2)

            # Quotes
            (c1,) = run(pc_text_encode, clip, 'Text saying "DOG MASK AND CAT COUPLE MASK(X)"')
            (c2,) = run(comfy, clip, 'Text saying "DOG MASK AND CAT COUPLE MASK(X)"')
            cond_equal(c1, c2)

            # Function cornercase
            (c1,) = run(pc_text_encode, clip, "test SDXL function")
            (c2,) = run(comfy, clip, "test SDXL function")
            (c3,) = run(pc_text_encode, clip, "test SDXL() function")
            cond_equal(c1, c2)

            # Weights
            (c1,) = run(pc_text_encode, clip, "(test:1.2) (test:0.6)")
            (c2,) = run(comfy, clip, "(test:1.2) (test:0.6)")
            cond_equal(c1, c2)

            # Concat
            (c1,) = run(pc_text_encode, clip, "test CAT test")
            (c2,) = run(concat, c, c)
            cond_equal(c1, c2)

            # Combine
            (c1,) = run(pc_text_encode, clip, "test AND test")
            (c2,) = run(combine, c, c)
            cond_equal(c1, c2)

            # Zero out
            (c1,) = run(pc_text_encode, clip, "test TE_WEIGHT(all=0)")
            (c2,) = run(zeroout, c)
            cond_equal(c1, c2)

            # Average
            (c1,) = run(comfy, clip, "test1")
            (c2,) = run(comfy, clip, "test2")
            (c3,) = run(pc_text_encode, clip, "test1 AVG() test2")
            (c4,) = run(pc_text_encode, clip, "test1 AVG test2")
            (avg,) = run(average, c1, c2, 0.5)
            cond_equal(avg, c3)
            cond_equal(avg, c4)

    @pytest.mark.xfail
    def test_failure(self, text_encoder_clips, pc_text_encode, node_class_objs):
        comfy = node_class_objs["comfy"]
        for _k, clip in text_encoder_clips:
            (c1,) = run(comfy, clip, "test SDXL function")
            (c2,) = run(pc_text_encode, clip, "test SDXL() function")
            cond_equal(c1, c2)

    def test_weight(self, text_encoder_clips, pc_text_encode, node_class_objs):
        comfy = node_class_objs["comfy"]
        combine = node_class_objs["combine"]
        strength = node_class_objs["strength"]

        for _k, clip in text_encoder_clips:
            (c,) = run(comfy, clip, "test")
            (c2,) = run(strength, c, 0.5)
            # Conditioning weights
            (a,) = run(pc_text_encode, clip, "test :0.5 AND test :0.5")
            (b,) = run(combine, c2, c2)
            cond_equal(a, b)
            cond_equal(a, b, "strength")
            # Weight == 0
            (a,) = run(pc_text_encode, clip, "test :0.5 AND test :0 AND test")
            (b,) = run(combine, c2, c)
            cond_equal(a, b)
            cond_equal(a, b, "strength")

    def test_attn_couple(self, text_encoder_clips, pc_text_encode):
        for _k, clip in text_encoder_clips:
            (c,) = run(pc_text_encode, clip, "test COUPLE prompt1 AND test2 COUPLE prompt2")
            (c2,) = run(pc_text_encode, clip, "test COUPLE prompt1 COUPLE test2 COUPLE prompt2")
            assert len(c) == 2
            assert len(c2) == 1

    def test_styles(self, text_encoder_clips, pc_text_encode, node_class_objs):
        comfy = node_class_objs["comfy"]
        for _k, clip in text_encoder_clips:
            (no_weights,) = run(comfy, clip, "this prompt has no weights")
            for style in ["comfy", "A1111", "comfy++", "compel", "down_weight", "perp"]:
                # no weights equal comfy
                (c,) = run(pc_text_encode, clip, "this prompt has no weights")
                cond_equal(no_weights, c)
                # does not fail when encoding weights
                for normalization in ["none", "mean", "length", "mean+length", "length+mean"]:
                    run(
                        pc_text_encode,
                        clip,
                        f"STYLE({style}, {normalization}) (this prompt) (has weights:0.9), (a:1.2) (b:1.2)",
                    )
                    # Just checking for exceptions

    def test_masks(self, text_encoder_clips, pc_text_encode, node_class_objs):
        comfy = node_class_objs["comfy"]
        solidmask = node_class_objs["solidmask"]
        setmask = node_class_objs["setmask"]
        for _k, clip in text_encoder_clips:
            (c1,) = run(pc_text_encode, clip, "test MASK()")
            (c2,) = run(comfy, clip, "test")
            (c2,) = run(setmask, c2, run(solidmask, 1.0, 512, 512)[0], "default", 1.0)
            cond_equal(c1, c2)
            cond_equal(c1, c2, "mask", tensors_equal)
