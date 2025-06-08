import unittest
import unittest.mock as mock
import logging

log = logging.getLogger("comfyui-prompt-control")


def reset_graphbuilder_state():
    from comfy_execution.graph_utils import GraphBuilder

    GraphBuilder.set_default_prefix("UID", 0, 0)


def find_file(name):
    names = {"test": "test.safetensors", "other": "some/other.safetensors"}
    return names.get(name)


def loraloader(text, adv=False, **kwargs):
    from .nodes_lazy import PCLazyLoraLoader, PCLazyLoraLoaderAdvanced

    reset_graphbuilder_state()
    if adv:
        cls = PCLazyLoraLoader
    else:
        cls = PCLazyLoraLoaderAdvanced
    model = [0, 1]
    clip = [0, 0]
    return cls().apply(unique_id="UID", model=model, clip=clip, text=text, **kwargs)


def te(text, adv=False, **kwargs):
    from .nodes_lazy import PCLazyTextEncode, PCLazyTextEncodeAdvanced

    if adv:
        cls = PCLazyTextEncode
    else:
        cls = PCLazyTextEncodeAdvanced
    reset_graphbuilder_state()
    clip = [0, 0]
    return cls().apply(clip=clip, text=text, unique_id="UID", **kwargs)


@mock.patch("prompt_control.utils.lora_name_to_file", find_file)
@mock.patch("torch.cuda.current_device", lambda: "cpu")
class GraphTests(unittest.TestCase):
    maxDiff = 4096

    def test_textencode(self):
        for p in ["test", "[test:0.2] test", "[test[test::0.5]]<lora:test:1>"]:
            r1 = te(p)
            r2 = te(p, adv=True)
            with self.subTest(f"Expansion: {p}"):
                self.assertEqual(r1, r2)

        reset_graphbuilder_state()
        with self.subTest("Expansion: LoRA"):
            r = te("test<lora:test:1>")
            self.assertEqual(
                r,
                {
                    "result": (["UID.0.0.2", 0],),
                    "expand": {
                        "UID.0.0.1": {"class_type": "PCTextEncode", "inputs": {"clip": [0, 0], "text": "test"}},
                        "UID.0.0.2": {
                            "class_type": "ConditioningSetTimestepRange",
                            "inputs": {"conditioning": ["UID.0.0.1", 0], "start": 0.0, "end": 1.0},
                        },
                    },
                },
            )
        with self.subTest("Expansion: LoRA with schedule"):
            r = te("simple [test:0.1,0.5] prompt<lora:test:1>")
            self.assertEqual(
                r,
                {
                    "result": (["UID.0.0.8", 0],),
                    "expand": {
                        "UID.0.0.1": {
                            "class_type": "PCTextEncode",
                            "inputs": {"clip": [0, 0], "text": "simple  prompt"},
                        },
                        "UID.0.0.2": {
                            "class_type": "ConditioningSetTimestepRange",
                            "inputs": {"conditioning": ["UID.0.0.1", 0], "start": 0.0, "end": 0.1},
                        },
                        "UID.0.0.3": {
                            "class_type": "PCTextEncode",
                            "inputs": {"clip": [0, 0], "text": "simple test prompt"},
                        },
                        "UID.0.0.4": {
                            "class_type": "ConditioningSetTimestepRange",
                            "inputs": {"conditioning": ["UID.0.0.3", 0], "start": 0.1, "end": 0.5},
                        },
                        "UID.0.0.5": {
                            "class_type": "PCTextEncode",
                            "inputs": {"clip": [0, 0], "text": "simple  prompt"},
                        },
                        "UID.0.0.6": {
                            "class_type": "ConditioningSetTimestepRange",
                            "inputs": {"conditioning": ["UID.0.0.5", 0], "start": 0.5, "end": 1.0},
                        },
                        "UID.0.0.7": {
                            "class_type": "ConditioningCombine",
                            "inputs": {"conditioning_1": ["UID.0.0.2", 0], "conditioning_2": ["UID.0.0.4", 0]},
                        },
                        "UID.0.0.8": {
                            "class_type": "ConditioningCombine",
                            "inputs": {"conditioning_1": ["UID.0.0.7", 0], "conditioning_2": ["UID.0.0.6", 0]},
                        },
                    },
                },
            )

    @mock.patch("prompt_control.utils.lora_name_to_file", find_file)
    def test_loraloader(self):
        with self.assertLogs(log, level="WARNING") as cm:
            result = loraloader("prompt here <lora:nonexistent:1.0:0.5>")["expand"]
            result_adv = loraloader("prompt here <lora:nonexistent:1.0:0.5>", adv=True)["expand"]
        self.assertIn("LoRA 'nonexistent' not found", cm.output[0])
        self.assertEqual(result, {})
        self.assertEqual(result_adv, {})

        result = loraloader("<lora:test:1>")["expand"]
        result2 = loraloader("prompt here <lora:test:1.0:0.5><lora:test:0:0.5>")["expand"]
        result3 = loraloader("prompt here <lora:test:1.0:0.5><lora:test:0:0.5>", adv=True)["expand"]
        self.assertEqual(result, result2)
        self.assertEqual(result2, result3)
        self.assertEqual(
            result,
            {
                "UID.0.0.1": {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": [0, 1],
                        "clip": [0, 0],
                        "strength_model": 1.0,
                        "strength_clip": 1.0,
                        "lora_name": "test.safetensors",
                    },
                }
            },
        )
        result = loraloader("<lora:test:1><lora:other:0.5>")["expand"]
        self.assertEqual(
            result,
            {
                "UID.0.0.1": {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": [0, 1],
                        "clip": [0, 0],
                        "strength_model": 1.0,
                        "strength_clip": 1.0,
                        "lora_name": "test.safetensors",
                    },
                },
                "UID.0.0.2": {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": ["UID.0.0.1", 0],
                        "clip": ["UID.0.0.1", 1],
                        "strength_model": 0.5,
                        "strength_clip": 0.5,
                        "lora_name": "some/other.safetensors",
                    },
                },
            },
        )

        result = loraloader("prompt here <lora:test:1.0:0.5>")["expand"]
        self.assertEqual(
            result,
            {
                "UID.0.0.1": {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": [0, 1],
                        "clip": [0, 0],
                        "strength_model": 1.0,
                        "strength_clip": 0.5,
                        "lora_name": "test.safetensors",
                    },
                }
            },
        )

        result = loraloader("prompt [<lora:test:0.5>:0.5]")["expand"]
        result2 = loraloader("prompt [<lora:test:0.5>:0.5]", adv=True)["expand"]
        self.assertEqual(result, result2)
        expected = {
            "UID.0.0.1": {
                "class_type": "CreateHookLora",
                "inputs": {"lora_name": "test.safetensors", "strength_model": 0.5, "strength_clip": 0.5},
            },
            "UID.0.0.2": {
                "class_type": "CreateHookKeyframe",
                "inputs": {"strength_mult": 0.0, "start_percent": 0.0},
            },
            "UID.0.0.3": {
                "class_type": "CreateHookKeyframe",
                "inputs": {
                    "start_percent": 0.5,
                    "prev_hook_kf": ["UID.0.0.2", 0],
                    "strength_mult": 1.0,
                },
            },
            "UID.0.0.4": {
                "class_type": "SetHookKeyframes",
                "inputs": {"hooks": ["UID.0.0.1", 0], "hook_kf": ["UID.0.0.3", 0]},
            },
            "UID.0.0.5": {
                "class_type": "SetClipHooks",
                "inputs": {
                    "clip": [0, 0],
                    "hooks": ["UID.0.0.4", 0],
                    "apply_to_conds": True,
                    "schedule_clip": True,
                },
            },
        }
        self.assertEqual(result, expected)
        result2 = loraloader("prompt [<lora:test:0.5>:0.5]", adv=True, start=0.6)["expand"]
        self.assertEqual(
            result2,
            {
                "UID.0.0.1": {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": [0, 1],
                        "clip": [0, 0],
                        "strength_model": 0.5,
                        "strength_clip": 0.5,
                        "lora_name": "test.safetensors",
                    },
                }
            },
        )
        result2 = loraloader("prompt [<lora:test:0.5>:0.5]", end=0.5)["expand"]
        self.assertEqual(result2, {})


if __name__ == "__main__":
    unittest.main()
