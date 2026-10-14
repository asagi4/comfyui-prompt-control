import unittest
from unittest.mock import patch

from prompt_control import prompts


class TestMissingIMaskFeather(unittest.TestCase):
    def test_missing_imask_does_not_shift_feather_to_next_mask(self):
        feather_sizes = []

        def record_feather(node, mask, left, top, right, bottom):
            self.assertIs(node, prompts.FeatherMask)
            feather_sizes.append(left)
            return (mask,)

        with patch.object(prompts, "call_node", side_effect=record_feather):
            _, mask, _ = prompts.get_mask("IMASK(1) FEATHER(2) IMASK(0) FEATHER(4)", (4, 4), [2.0])

        self.assertEqual(feather_sizes, [4])
        self.assertEqual(mask, 2.0)

    def test_missing_only_imask_never_feathers_none(self):
        with (
            patch.object(prompts, "call_node", side_effect=AssertionError("cannot feather a missing mask")) as call,
            self.assertLogs("comfyui-prompt-control", level="WARNING"),
        ):
            _, mask, _ = prompts.get_mask("IMASK(0) FEATHER(4)", (4, 4), None)

        self.assertIsNone(mask)
        call.assert_not_called()
