import unittest
import unittest.mock as mock


def find_file(name):
    names = {"test": "test.safetensors", "other": "some/other.safetensors"}
    return names.get(name)


class GraphTests(unittest.TestCase):
    maxDiff = 4096

    @mock.patch("prompt_control.utils.lora_name_to_file", find_file)
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
                "result": (["UID2", 0],),
                "expand": {
                    "UID1": {"class_type": "PCTextEncode", "inputs": {"clip": [0, 0], "text": "test"}},
                    "UID2": {
                        "class_type": "ConditioningSetTimestepRange",
                        "inputs": {"conditioning": ["UID1", 0], "start": 0.0, "end": 1.0},
                    },
                },
            },
        )
        r = PCLazyTextEncode().apply(clip, "simple [test:0.1,0.5] prompt<lora:test:1>", "UID")
        self.assertEqual(
            r,
            {
                "result": (["UID8", 0],),
                "expand": {
                    "UID1": {"class_type": "PCTextEncode", "inputs": {"clip": [0, 0], "text": "simple  prompt"}},
                    "UID2": {
                        "class_type": "ConditioningSetTimestepRange",
                        "inputs": {"conditioning": ["UID1", 0], "start": 0.0, "end": 0.1},
                    },
                    "UID3": {"class_type": "PCTextEncode", "inputs": {"clip": [0, 0], "text": "simple test prompt"}},
                    "UID4": {
                        "class_type": "ConditioningSetTimestepRange",
                        "inputs": {"conditioning": ["UID3", 0], "start": 0.1, "end": 0.5},
                    },
                    "UID5": {"class_type": "PCTextEncode", "inputs": {"clip": [0, 0], "text": "simple  prompt"}},
                    "UID6": {
                        "class_type": "ConditioningSetTimestepRange",
                        "inputs": {"conditioning": ["UID5", 0], "start": 0.5, "end": 1.0},
                    },
                    "UID7": {
                        "class_type": "ConditioningCombine",
                        "inputs": {"conditioning_1": ["UID2", 0], "conditioning_2": ["UID4", 0]},
                    },
                    "UID8": {
                        "class_type": "ConditioningCombine",
                        "inputs": {"conditioning_1": ["UID7", 0], "conditioning_2": ["UID6", 0]},
                    },
                },
            },
        )

    @mock.patch("prompt_control.utils.lora_name_to_file", find_file)
    def test_loraloader(self):
        from .nodes_lazy import PCLazyLoraLoader, PCLazyLoraLoaderAdvanced

        model = [0, 1]
        clip = [0, 0]
        with self.assertLogs("comfyui-prompt-control", level="WARNING") as cm:
            result = PCLazyLoraLoader().apply(model, clip, "prompt here <lora:nonexistent:1.0:0.5>", "UID")["expand"]
            result_adv = PCLazyLoraLoaderAdvanced().apply(model, clip, "prompt here <lora:nonexistent:1.0:0.5>", "UID")[
                "expand"
            ]
        self.assertIn("LoRA 'nonexistent' not found", cm.output[0])
        self.assertIn("LoRA 'nonexistent' not found", cm.output[1])
        self.assertEqual(result, {})
        self.assertEqual(result_adv, {})

        result = PCLazyLoraLoader().apply(model, clip, "<lora:test:1>", "UID")["expand"]
        result2 = PCLazyLoraLoader().apply(model, clip, "prompt here <lora:test:1.0:0.5><lora:test:0:0.5>", "UID")[
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
                "UID1": {
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
        result = PCLazyLoraLoader().apply(model, clip, "<lora:test:1><lora:other:0.5>", "UID")["expand"]
        self.assertEqual(
            result,
            {
                "UID1": {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": [0, 1],
                        "clip": [0, 0],
                        "strength_model": 1.0,
                        "strength_clip": 1.0,
                        "lora_name": "test.safetensors",
                    },
                },
                "UID2": {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": ["UID1", 0],
                        "clip": ["UID1", 1],
                        "strength_model": 0.5,
                        "strength_clip": 0.5,
                        "lora_name": "some/other.safetensors",
                    },
                },
            },
        )

        result = PCLazyLoraLoader().apply(model, clip, "prompt here <lora:test:1.0:0.5>", "UID")["expand"]
        self.assertEqual(
            result,
            {
                "UID1": {
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

        result = PCLazyLoraLoader().apply(model, clip, "prompt [<lora:test:0.5>:0.5]", "UID")["expand"]
        result2 = PCLazyLoraLoaderAdvanced().apply(model, clip, "prompt [<lora:test:0.5>:0.5]", "UID")["expand"]
        self.assertEqual(result, result2)
        expected = {
            "UID1": {
                "class_type": "CreateHookLora",
                "inputs": {"lora_name": "test.safetensors", "strength_model": 0.5, "strength_clip": 0.5},
            },
            "UID2": {
                "class_type": "CreateHookKeyframe",
                "inputs": {"strength_mult": 0.0, "start_percent": 0.0},
            },
            "UID3": {
                "class_type": "CreateHookKeyframe",
                "inputs": {
                    "start_percent": 0.5,
                    "prev_hook_kf": ["UID2", 0],
                    "strength_mult": 1.0,
                },
            },
            "UID4": {
                "class_type": "SetHookKeyframes",
                "inputs": {"hooks": ["UID1", 0], "hook_kf": ["UID3", 0]},
            },
            "UID5": {
                "class_type": "SetClipHooks",
                "inputs": {
                    "clip": [0, 0],
                    "hooks": ["UID4", 0],
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
                "UID1": {
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
