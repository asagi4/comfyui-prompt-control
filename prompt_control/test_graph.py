import unittest
import unittest.mock as mock
import logging

log = logging.getLogger("comfyui-prompt-control")


def find_file(name):
    names = {"test": "test.safetensors", "other": "some/other.safetensors"}
    return names.get(name)


@mock.patch("prompt_control.utils.lora_name_to_file", find_file)
@mock.patch.dict("sys.modules", nodes=mock.MagicMock())
class GraphTests(unittest.TestCase):
    maxDiff = 4096

    def test_textencode(self):
        clip = [0, 0]
        from .nodes_lazy import PCLazyTextEncode, PCLazyTextEncodeAdvanced

        for p in ["test", "[test:0.2] test", "[test[test::0.5]]<lora:test:1>"]:
            r1 = PCLazyTextEncode().apply(clip, p, "UID")
            r2 = PCLazyTextEncodeAdvanced().apply(clip, p, "UID")
            self.assertEqual(r1, r2)

        r = PCLazyTextEncode().apply(clip, "test<lora:test:1>", "UID")
        self.assertEqual(
            r,
            {
                "result": (["UID-2", 0],),
                "expand": {
                    "UID-1": {"class_type": "PCTextEncode", "inputs": {"clip": [0, 0], "text": "test"}},
                    "UID-2": {
                        "class_type": "ConditioningSetTimestepRange",
                        "inputs": {"conditioning": ["UID-1", 0], "start": 0.0, "end": 1.0},
                    },
                },
            },
        )
        r = PCLazyTextEncode().apply(clip, "simple [test:0.1,0.5] prompt<lora:test:1>", "UID")
        self.assertEqual(
            r,
            {
                "result": (["UID-8", 0],),
                "expand": {
                    "UID-1": {"class_type": "PCTextEncode", "inputs": {"clip": [0, 0], "text": "simple  prompt"}},
                    "UID-2": {
                        "class_type": "ConditioningSetTimestepRange",
                        "inputs": {"conditioning": ["UID-1", 0], "start": 0.0, "end": 0.1},
                    },
                    "UID-3": {"class_type": "PCTextEncode", "inputs": {"clip": [0, 0], "text": "simple test prompt"}},
                    "UID-4": {
                        "class_type": "ConditioningSetTimestepRange",
                        "inputs": {"conditioning": ["UID-3", 0], "start": 0.1, "end": 0.5},
                    },
                    "UID-5": {"class_type": "PCTextEncode", "inputs": {"clip": [0, 0], "text": "simple  prompt"}},
                    "UID-6": {
                        "class_type": "ConditioningSetTimestepRange",
                        "inputs": {"conditioning": ["UID-5", 0], "start": 0.5, "end": 1.0},
                    },
                    "UID-7": {
                        "class_type": "ConditioningCombine",
                        "inputs": {"conditioning_1": ["UID-2", 0], "conditioning_2": ["UID-4", 0]},
                    },
                    "UID-8": {
                        "class_type": "ConditioningCombine",
                        "inputs": {"conditioning_1": ["UID-7", 0], "conditioning_2": ["UID-6", 0]},
                    },
                },
            },
        )

    @mock.patch("prompt_control.utils.lora_name_to_file", find_file)
    def test_loraloader(self):
        from .nodes_lazy import PCLazyLoraLoader, PCLazyLoraLoaderAdvanced

        model = [0, 1]
        clip = [0, 0]
        with self.assertLogs(log, level="WARNING") as cm:
            result = PCLazyLoraLoader().apply("UID", model, clip, "prompt here <lora:nonexistent:1.0:0.5>")["expand"]
            result_adv = PCLazyLoraLoaderAdvanced().apply(model, clip, "prompt here <lora:nonexistent:1.0:0.5>", "UID")[
                "expand"
            ]
        self.assertIn("LoRA 'nonexistent' not found", cm.output[0])
        self.assertEqual(result, {})
        self.assertEqual(result_adv, {})

        result = PCLazyLoraLoader().apply("UID", model, clip, "<lora:test:1>")["expand"]
        result2 = PCLazyLoraLoader().apply("UID", model, clip, "prompt here <lora:test:1.0:0.5><lora:test:0:0.5>")[
            "expand"
        ]
        result3 = PCLazyLoraLoaderAdvanced().apply(
            model, clip, "prompt here <lora:test:1.0:0.5><lora:test:0:0.5>", "UID"
        )["expand"]
        self.assertEqual(result, result2)
        self.assertEqual(result2, result3)
        self.assertEqual(
            result,
            {
                "UID-1": {
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
        result = PCLazyLoraLoader().apply("UID", model, clip, "<lora:test:1><lora:other:0.5>")["expand"]
        self.assertEqual(
            result,
            {
                "UID-1": {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": [0, 1],
                        "clip": [0, 0],
                        "strength_model": 1.0,
                        "strength_clip": 1.0,
                        "lora_name": "test.safetensors",
                    },
                },
                "UID-2": {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": ["UID-1", 0],
                        "clip": ["UID-1", 1],
                        "strength_model": 0.5,
                        "strength_clip": 0.5,
                        "lora_name": "some/other.safetensors",
                    },
                },
            },
        )

        result = PCLazyLoraLoader().apply("UID", model, clip, "prompt here <lora:test:1.0:0.5>")["expand"]
        self.assertEqual(
            result,
            {
                "UID-1": {
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

        result = PCLazyLoraLoader().apply("UID", model, clip, "prompt [<lora:test:0.5>:0.5]")["expand"]
        result2 = PCLazyLoraLoaderAdvanced().apply(model, clip, "prompt [<lora:test:0.5>:0.5]", "UID")["expand"]
        self.assertEqual(result, result2)
        expected = {
            "UID-1": {
                "class_type": "CreateHookLora",
                "inputs": {"lora_name": "test.safetensors", "strength_model": 0.5, "strength_clip": 0.5},
            },
            "UID-2": {
                "class_type": "CreateHookKeyframe",
                "inputs": {"strength_mult": 0.0, "start_percent": 0.0},
            },
            "UID-3": {
                "class_type": "CreateHookKeyframe",
                "inputs": {
                    "start_percent": 0.5,
                    "prev_hook_kf": ["UID-2", 0],
                    "strength_mult": 1.0,
                },
            },
            "UID-4": {
                "class_type": "SetHookKeyframes",
                "inputs": {"hooks": ["UID-1", 0], "hook_kf": ["UID-3", 0]},
            },
            "UID-5": {
                "class_type": "SetClipHooks",
                "inputs": {
                    "clip": [0, 0],
                    "hooks": ["UID-4", 0],
                    "apply_to_conds": True,
                    "schedule_clip": True,
                },
            },
        }
        self.assertEqual(result, expected)
        result2 = PCLazyLoraLoaderAdvanced().apply(model, clip, "prompt [<lora:test:0.5>:0.5]", "UID", start=0.6)[
            "expand"
        ]
        self.assertEqual(
            result2,
            {
                "UID-1": {
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
        result2 = PCLazyLoraLoaderAdvanced().apply(model, clip, "prompt [<lora:test:0.5>:0.5]", "UID", end=0.5)[
            "expand"
        ]
        self.assertEqual(result2, {})


if __name__ == "__main__":
    unittest.main()
