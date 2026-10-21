import unittest

import comfy.conds
import torch
from comfy.model_base import Anima

from prompt_control import anima_mix


def cond(value, ids):
    return [
        torch.full((1, 2, 3), float(value)),
        {"t5xxl_ids": torch.tensor(ids), "t5xxl_weights": torch.ones(len(ids))},
    ]


class TestAvgFractions(unittest.TestCase):
    def test_left_fold(self):
        # p1 AVG(0.5) p2 AVG(0.5) p3 -> 0.25, 0.25, 0.5
        self.assertEqual(anima_mix.avg_fractions([0.5, 0.5, 1.0]), [0.25, 0.25, 0.5])
        # PLv4's 1:3 -> AVG(0.25)
        self.assertEqual(anima_mix.avg_fractions([0.25, 1.0]), [0.25, 0.75])

    def test_weight_one_drops_the_next_entry(self):
        self.assertEqual(anima_mix.avg_fractions([1.0, 1.0]), [1.0, 0.0])


class TestAttach(unittest.TestCase):
    def test_avg_entries_keep_their_own_ids(self):
        a, b = [cond(1, [1, 2])], [cond(3, [7, 8, 9])]
        result = [[torch.full((1, 2, 3), 2.0), dict(a[0][1])]]
        anima_mix.attach(result, [(anima_mix.cat_parts([a]), 0.5), (anima_mix.cat_parts([b]), 1.0)])
        mix = result[0][1][anima_mix.MIX_KEY]
        self.assertEqual([f for f, _ in mix], [0.5, 0.5])
        self.assertEqual([pieces[0][1].tolist() for _, pieces in mix], [[1, 2], [7, 8, 9]])
        self.assertNotIn(anima_mix.MIX_KEY, a[0][1])  # the input dict is not mutated

    def test_cat_alone_is_recorded(self):
        a, b = [cond(1, [1, 2])], [cond(3, [7])]
        parts = anima_mix.cat_parts([a, b])
        a[0][0] = torch.cat((a[0][0], b[0][0]), 1)  # what the concat loop does in place
        result = [list(a[0])]
        anima_mix.attach(result, [(parts, 1.0)])
        ((fraction, pieces),) = result[0][1][anima_mix.MIX_KEY]
        self.assertEqual(fraction, 1.0)
        self.assertEqual([p[0].shape[1] for p in pieces], [2, 2])  # the pre-concat tensors
        self.assertEqual([p[1].tolist() for p in pieces], [[1, 2], [7]])

    def test_single_plain_prompt_is_untouched(self):
        a = [cond(1, [1, 2])]
        result = [list(a[0])]
        anima_mix.attach(result, [(anima_mix.cat_parts([a]), 1.0)])
        self.assertNotIn(anima_mix.MIX_KEY, result[0][1])

    def test_non_anima_conds_are_untouched(self):
        a = [[torch.zeros((1, 2, 3)), {}]]
        b = [[torch.ones((1, 2, 3)), {}]]
        result = [[torch.zeros((1, 2, 3)), {}]]
        anima_mix.attach(result, [(anima_mix.cat_parts([a]), 0.5), (anima_mix.cat_parts([b]), 1.0)])
        self.assertEqual(result[0][1], {})


class TestExtraConds(unittest.TestCase):
    def setUp(self):
        self.saved = Anima.extra_conds

        # Stand-in for the real adapter + Anima's padding: one row per id, padded to 4.
        def fake(self, **kwargs):
            ids = kwargs["t5xxl_ids"]
            out = kwargs["cross_attn"].mean() + ids.float().view(1, -1, 1).expand(1, -1, 3)
            out = torch.nn.functional.pad(out, (0, 0, 0, max(0, 4 - out.shape[1])))
            return {"c_crossattn": comfy.conds.CONDRegular(out), "other": kwargs.get("other")}

        Anima.extra_conds = fake
        self.min_length = anima_mix.MIN_LENGTH
        anima_mix.MIN_LENGTH = 4
        anima_mix.install()

    def tearDown(self):
        Anima.extra_conds = self.saved
        anima_mix.MIN_LENGTH = self.min_length

    def run_mix(self, mix):
        h = torch.zeros((1, 2, 3))
        with torch.inference_mode():
            return Anima.extra_conds(
                None, cross_attn=h, t5xxl_ids=torch.tensor([0]), other=5, **{anima_mix.MIX_KEY: mix}
            )

    def test_avg_mixes_adapter_outputs(self):
        h1, h2 = torch.zeros((1, 2, 3)), torch.ones((1, 2, 3))
        out = self.run_mix(
            [(0.25, [(h1, torch.tensor([10, 10]), None)]), (0.75, [(h2, torch.tensor([20, 20, 20]), None)])]
        )
        mixed = out["c_crossattn"].cond
        self.assertEqual(mixed.shape, (1, 4, 3))
        self.assertTrue(torch.allclose(mixed[0, :2], torch.full((2, 3), 0.25 * 10 + 0.75 * 21)))
        self.assertTrue(torch.allclose(mixed[0, 2], torch.full((3,), 0.75 * 21)))  # shorter entry zero-padded
        self.assertTrue(torch.equal(mixed[0, 3], torch.zeros(3)))
        self.assertEqual(out["other"], 5)

    def test_avg_is_order_independent(self):
        e1 = (0.5, [(torch.zeros((1, 2, 3)), torch.tensor([1, 2]), None)])
        e2 = (0.5, [(torch.ones((1, 2, 3)), torch.tensor([3, 4, 5]), None)])
        self.assertTrue(
            torch.equal(self.run_mix([e1, e2])["c_crossattn"].cond, self.run_mix([e2, e1])["c_crossattn"].cond)
        )

    def test_cat_joins_unpadded_pieces(self):
        h = torch.zeros((1, 2, 3))
        out = self.run_mix([(1.0, [(h, torch.tensor([1, 2]), None), (h, torch.tensor([7]), None)])])
        rows = out["c_crossattn"].cond[0, :, 0].tolist()
        self.assertEqual(rows, [1.0, 2.0, 7.0, 0.0])  # no padding between the pieces, padded at the end

    def test_cat_drops_inner_eos_only(self):
        h = torch.zeros((1, 2, 3))
        eos = anima_mix.T5_EOS
        out = self.run_mix([(1.0, [(h, torch.tensor([5, eos]), None), (h, torch.tensor([7, eos]), None)])])
        self.assertEqual(out["c_crossattn"].cond[0, :, 0].tolist(), [5.0, 7.0, float(eos), 0.0])

    def test_cat_longer_than_min_is_not_truncated(self):
        h = torch.zeros((1, 2, 3))
        out = self.run_mix([(1.0, [(h, torch.tensor([1, 2, 3]), None), (h, torch.tensor([4, 5]), None)])])
        self.assertEqual(out["c_crossattn"].cond[0, :, 0].tolist(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_plain_cond_takes_original_path(self):
        with torch.inference_mode():
            out = Anima.extra_conds(None, cross_attn=torch.zeros((1, 2, 3)), t5xxl_ids=torch.tensor([4, 4]))
        self.assertTrue(torch.equal(out["c_crossattn"].cond[:, :2], torch.full((1, 2, 3), 4.0)))

    def test_install_is_idempotent(self):
        wrapped = Anima.extra_conds
        anima_mix.install()
        self.assertIs(Anima.extra_conds, wrapped)
