import logging

import pytest
from comfy_execution.graph_utils import GraphBuilder

from prompt_control.nodes_lazy import (
    PCLazyLoraLoader,
    PCLazyLoraLoaderAdvanced,
    PCLazyTextEncode,
    PCLazyTextEncodeAdvanced,
)

log = logging.getLogger("comfyui-prompt-control")


def reset_graphbuilder_state():
    GraphBuilder.set_default_prefix("UID", 0, 0)


def find_file(name):
    names = {"test": "test.safetensors", "other": "some/other.safetensors"}
    return names.get(name)


def loraloader(text, adv=False, **kwargs):
    reset_graphbuilder_state()
    cls = PCLazyLoraLoader if adv else PCLazyLoraLoaderAdvanced
    model = [0, 1]
    clip = [0, 0]
    return cls().apply(unique_id="UID", model=model, clip=clip, text=text, **kwargs)


def te(text, adv=False, **kwargs):
    cls = PCLazyTextEncode if adv else PCLazyTextEncodeAdvanced
    reset_graphbuilder_state()
    clip = [0, 0]
    return cls().apply(clip=clip, text=text, unique_id="UID", **kwargs)


@pytest.fixture(autouse=True)
def patch_lora_name_to_file(monkeypatch):
    import prompt_control.utils

    monkeypatch.setattr(prompt_control.utils, "lora_name_to_file", find_file)


@pytest.fixture(autouse=True)
def patch_torch_cuda_current_device(monkeypatch):
    import torch.cuda

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")


def test_textencode_expansion():
    for p in ["test", "[test:0.2] test", "[test[test::0.5]]<lora:test:1>"]:
        r1 = te(p)
        r2 = te(p, adv=True)
        assert r1 == r2


def test_textencode_alternating():
    r = te("[a|b]")
    expected_result = {
        "expand": {
            "UID.0.0.1": {
                "class_type": "PCTextEncode",
                "inputs": {
                    "clip": [
                        0,
                        0,
                    ],
                    "text": "a",
                },
            },
            "UID.0.0.10": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {
                    "conditioning": [
                        "UID.0.0.9",
                        0,
                    ],
                    "end": 0.5,
                    "start": 0.4,
                },
            },
            "UID.0.0.11": {
                "class_type": "PCTextEncode",
                "inputs": {
                    "clip": [
                        0,
                        0,
                    ],
                    "text": "b",
                },
            },
            "UID.0.0.12": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {
                    "conditioning": [
                        "UID.0.0.11",
                        0,
                    ],
                    "end": 0.6,
                    "start": 0.5,
                },
            },
            "UID.0.0.13": {
                "class_type": "PCTextEncode",
                "inputs": {
                    "clip": [
                        0,
                        0,
                    ],
                    "text": "a",
                },
            },
            "UID.0.0.14": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {
                    "conditioning": [
                        "UID.0.0.13",
                        0,
                    ],
                    "end": 0.7,
                    "start": 0.6,
                },
            },
            "UID.0.0.15": {
                "class_type": "PCTextEncode",
                "inputs": {
                    "clip": [
                        0,
                        0,
                    ],
                    "text": "b",
                },
            },
            "UID.0.0.16": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {
                    "conditioning": [
                        "UID.0.0.15",
                        0,
                    ],
                    "end": 0.8,
                    "start": 0.7,
                },
            },
            "UID.0.0.17": {
                "class_type": "PCTextEncode",
                "inputs": {
                    "clip": [
                        0,
                        0,
                    ],
                    "text": "a",
                },
            },
            "UID.0.0.18": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {
                    "conditioning": [
                        "UID.0.0.17",
                        0,
                    ],
                    "end": 0.9,
                    "start": 0.8,
                },
            },
            "UID.0.0.19": {
                "class_type": "PCTextEncode",
                "inputs": {
                    "clip": [
                        0,
                        0,
                    ],
                    "text": "b",
                },
            },
            "UID.0.0.2": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {
                    "conditioning": [
                        "UID.0.0.1",
                        0,
                    ],
                    "end": 0.1,
                    "start": 0.0,
                },
            },
            "UID.0.0.20": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {
                    "conditioning": [
                        "UID.0.0.19",
                        0,
                    ],
                    "end": 1.0,
                    "start": 0.9,
                },
            },
            "UID.0.0.21": {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [
                        "UID.0.0.2",
                        0,
                    ],
                    "conditioning_2": [
                        "UID.0.0.4",
                        0,
                    ],
                },
            },
            "UID.0.0.22": {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [
                        "UID.0.0.21",
                        0,
                    ],
                    "conditioning_2": [
                        "UID.0.0.6",
                        0,
                    ],
                },
            },
            "UID.0.0.23": {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [
                        "UID.0.0.22",
                        0,
                    ],
                    "conditioning_2": [
                        "UID.0.0.8",
                        0,
                    ],
                },
            },
            "UID.0.0.24": {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [
                        "UID.0.0.23",
                        0,
                    ],
                    "conditioning_2": [
                        "UID.0.0.10",
                        0,
                    ],
                },
            },
            "UID.0.0.25": {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [
                        "UID.0.0.24",
                        0,
                    ],
                    "conditioning_2": [
                        "UID.0.0.12",
                        0,
                    ],
                },
            },
            "UID.0.0.26": {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [
                        "UID.0.0.25",
                        0,
                    ],
                    "conditioning_2": [
                        "UID.0.0.14",
                        0,
                    ],
                },
            },
            "UID.0.0.27": {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [
                        "UID.0.0.26",
                        0,
                    ],
                    "conditioning_2": [
                        "UID.0.0.16",
                        0,
                    ],
                },
            },
            "UID.0.0.28": {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [
                        "UID.0.0.27",
                        0,
                    ],
                    "conditioning_2": [
                        "UID.0.0.18",
                        0,
                    ],
                },
            },
            "UID.0.0.29": {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [
                        "UID.0.0.28",
                        0,
                    ],
                    "conditioning_2": [
                        "UID.0.0.20",
                        0,
                    ],
                },
            },
            "UID.0.0.3": {
                "class_type": "PCTextEncode",
                "inputs": {
                    "clip": [
                        0,
                        0,
                    ],
                    "text": "b",
                },
            },
            "UID.0.0.4": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {
                    "conditioning": [
                        "UID.0.0.3",
                        0,
                    ],
                    "end": 0.2,
                    "start": 0.1,
                },
            },
            "UID.0.0.5": {
                "class_type": "PCTextEncode",
                "inputs": {
                    "clip": [
                        0,
                        0,
                    ],
                    "text": "a",
                },
            },
            "UID.0.0.6": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {
                    "conditioning": [
                        "UID.0.0.5",
                        0,
                    ],
                    "end": 0.3,
                    "start": 0.2,
                },
            },
            "UID.0.0.7": {
                "class_type": "PCTextEncode",
                "inputs": {
                    "clip": [
                        0,
                        0,
                    ],
                    "text": "b",
                },
            },
            "UID.0.0.8": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {
                    "conditioning": [
                        "UID.0.0.7",
                        0,
                    ],
                    "end": 0.4,
                    "start": 0.3,
                },
            },
            "UID.0.0.9": {
                "class_type": "PCTextEncode",
                "inputs": {
                    "clip": [
                        0,
                        0,
                    ],
                    "text": "a",
                },
            },
        },
        "result": (
            [
                "UID.0.0.29",
                0,
            ],
        ),
    }
    assert r == expected_result


def test_textencode_lora():
    reset_graphbuilder_state()
    r = te("test<lora:test:1>")
    assert r == {
        "result": (["UID.0.0.2", 0],),
        "expand": {
            "UID.0.0.1": {"class_type": "PCTextEncode", "inputs": {"clip": [0, 0], "text": "test"}},
            "UID.0.0.2": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {"conditioning": ["UID.0.0.1", 0], "start": 0.0, "end": 1.0},
            },
        },
    }


def test_textencode_lora_with_schedule():
    r = te("simple [test:0.1,0.5] prompt<lora:test:1>")
    assert r == {
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
    }


def test_loraloader_empty(monkeypatch, caplog):
    result = loraloader("prompt here <lora:nonexistent:1.0:0.5>")["expand"]
    result_adv = loraloader("prompt here <lora:nonexistent:1.0:0.5>", adv=True)["expand"]
    assert result == {}
    assert result_adv == {}


def test_loraloader_duplicate_results():
    result = loraloader("<lora:test:1>")["expand"]
    result2 = loraloader("prompt here <lora:test:1.0:0.5><lora:test:0:0.5>")["expand"]
    result3 = loraloader("prompt here <lora:test:1.0:0.5><lora:test:0:0.5>", adv=True)["expand"]
    assert result == result2
    assert result2 == result3
    assert result == {
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
    }


def test_loraloader_multiple_loras():
    result = loraloader("<lora:test:1><lora:other:0.5>")["expand"]
    assert result == {
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
    }


def test_loraloader_strength_clip():
    result = loraloader("prompt here <lora:test:1.0:0.5>")["expand"]
    assert result == {
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
    }


def test_loraloader_scheduled_compare():
    result = loraloader("prompt [<lora:test:0.5>:0.5]")["expand"]
    result2 = loraloader("prompt [<lora:test:0.5>:0.5]", adv=True)["expand"]
    assert result == result2
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
            "inputs": {"start_percent": 0.5, "prev_hook_kf": ["UID.0.0.2", 0], "strength_mult": 1.0},
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
    assert result == expected


def test_loraloader_adv_start():
    result2 = loraloader("prompt [<lora:test:0.5>:0.5]", adv=True, start=0.6)["expand"]
    assert result2 == {
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
    }


def test_loraloader_end_zero():
    result2 = loraloader("prompt [<lora:test:0.5>:0.5]", end=0.5)["expand"]
    assert result2 == {}
