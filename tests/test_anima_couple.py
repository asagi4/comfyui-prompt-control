import unittest
from types import SimpleNamespace

import torch
from comfy.hooks import HookGroup
from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
from comfy.model_base import Anima

from prompt_control import anima_couple
from prompt_control.attention_couple_ppm import AttentionCoupleHook


class TestAnimaAttentionHook(unittest.TestCase):
    def test_cosmos_patch_contract_and_masked_output(self):
        hook = AttentionCoupleHook()
        hook.num_conds = 2
        hook.base_strength = 1.0
        hook.mask = torch.tensor(
            [
                [[[1.0, 0.0]]],
                [[[0.0, 1.0]]],
            ]
        )

        base = torch.zeros((1, 3, 4))
        region = torch.ones((1, 3, 4))
        options = {
            "activations_shape": [1, 4, 1, 1, 2],
            "cond_or_uncond": [hook.COND],
            "pc_couple": {"processed_conds": [base, region]},
        }
        q = torch.zeros((1, 2, 4))
        k = torch.zeros((1, 3, 4))
        v = torch.zeros((1, 3, 4))
        pe = object()

        patched = hook.anima_attn2_patch(q, k, v, pe=pe, attn_mask=None, extra_options=options)

        self.assertEqual(set(patched), {"q", "k", "v", "pe", "attn_mask"})
        self.assertEqual(patched["q"].shape, (2, 2, 4))
        self.assertEqual(patched["k"].shape, (2, 3, 4))
        self.assertEqual(patched["v"].shape, (2, 3, 4))
        self.assertIs(patched["pe"], pe)

        def attention(_q, _k, _v, _heads, **_kwargs):
            return torch.tensor(
                [
                    [[10.0], [20.0]],
                    [[30.0], [40.0]],
                ]
            )

        out = hook.anima_optimized_attention(
            attention,
            patched["q"],
            patched["k"],
            patched["v"],
            1,
            transformer_options=options,
        )

        torch.testing.assert_close(out, torch.tensor([[[10.0], [40.0]]]))

    def test_anima_uses_cosmos_hooks_and_keeps_existing_attn2_patches(self):
        hook = AttentionCoupleHook()
        base = torch.zeros((1, 3, 4))
        region = torch.ones((1, 3, 4))
        hook.comfy_conds = [[base, {}], [region, {}]]
        hook.conds = [base, region]
        hook.num_conds = 2
        hook.mask = torch.ones((2, 1, 1, 2))

        existing_patch = object()
        options = {"patches": {"attn2_patch": [existing_patch]}}
        model = SimpleNamespace(model=object.__new__(Anima), model_options={})

        hook.add_hook_patches(model, {}, {}, HookGroup())
        hook.on_apply_hooks(model, options)

        self.assertEqual(len(options["patches"]["attn2_patch"]), 2)
        self.assertIs(options["patches"]["attn2_patch"][1], existing_patch)
        self.assertNotIn("attn2_output_patch", options["patches"])
        self.assertIn("optimized_attention_override", options)

    def test_forward_wrapper_does_not_patch_attention_modules(self):
        class RecordingExecutor:
            class_obj = SimpleNamespace(patch_spatial=2)

            def __call__(self, *args, **kwargs):
                return kwargs["transformer_options"]

        attn2_patches = [object()]
        options = {
            "pc_couple": {"processed_conds": []},
            "patches": {"attn2_patch": attn2_patches},
        }
        x = SimpleNamespace(shape=(1, 4, 1, 8, 8))

        forwarded = anima_couple.anima_forward_wrapper(
            RecordingExecutor(),
            x,
            transformer_options=options,
        )

        self.assertEqual(forwarded["activations_shape"], [1, 4, 1, 4, 4])
        self.assertIs(forwarded["patches"]["attn2_patch"], attn2_patches)

    def test_cosmos_attention_runs_with_anima_couple_hooks(self):
        hook = AttentionCoupleHook()
        hook.num_conds = 2
        hook.base_strength = 1.0
        hook.mask = torch.tensor(
            [
                [[[1.0, 0.0]]],
                [[[0.0, 1.0]]],
            ]
        )

        base = torch.zeros((1, 3, 4))
        region = torch.ones((1, 3, 4))
        options = {
            "activations_shape": [1, 4, 1, 1, 2],
            "cond_or_uncond": [hook.COND],
            "pc_couple": {"processed_conds": [base, region]},
            "patches": {"attn2_patch": [hook.anima_attn2_patch]},
            "optimized_attention_override": hook.anima_optimized_attention,
        }
        attention = CosmosAttention(
            query_dim=4,
            context_dim=4,
            n_heads=1,
            head_dim=4,
            operations=torch.nn,
        )

        out = attention(torch.zeros((1, 2, 4)), base, transformer_options=options)

        self.assertEqual(out.shape, (1, 2, 4))


if __name__ == "__main__":
    unittest.main()
