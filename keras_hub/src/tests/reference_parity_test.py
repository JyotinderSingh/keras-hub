import importlib
import os
import tempfile
from unittest import mock

import numpy as np
import pytest

from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.tests import reference_parity
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_gpt2

GPT2_CONFIG = {
    "vocab_size": 211,
    "n_positions": 64,
    "n_embd": 48,
    "n_layer": 3,
    "n_head": 4,
    # One id for both, as in GPT-2; the default is outside this vocabulary.
    "bos_token_id": 1,
    "eos_token_id": 1,
}


class ReferenceParityTest(TestCase):
    def test_unit_draw_fixed_value(self):
        # Every parity test depends on these values; change them on purpose.
        draw = reference_parity.unit_draw(0, "wte.weight", (2, 3))
        expected = [
            [-1.1977031545881198, 0.8892476811345476, -0.7633708311511904],
            [0.6460901334155383, -0.3194623459182225, 0.1709566494281259],
        ]
        self.assertAllEqual(draw, expected)

    def test_unit_draw_is_keyed(self):
        draw = reference_parity.unit_draw(0, "wte.weight", (100_000,))
        self.assertAllEqual(
            draw, reference_parity.unit_draw(0, "wte.weight", (100_000,))
        )
        other_seed = reference_parity.unit_draw(1, "wte.weight", (100_000,))
        other_key = reference_parity.unit_draw(0, "wpe.weight", (100_000,))
        self.assertFalse(np.array_equal(draw, other_seed))
        self.assertFalse(np.array_equal(draw, other_key))
        # The standard error of both moments is about 0.003.
        self.assertAllClose(draw.mean(), 0.0, atol=0.02)
        self.assertAllClose(draw.std(), 1.0, atol=0.02)

    def test_redraw_rules(self):
        pytest.importorskip("safetensors")
        from safetensors.numpy import save_file

        tensors = {
            "dense.weight": np.zeros((8, 4), dtype="float32"),
            "dense.bias": np.zeros((4,), dtype="float32"),
            "norm.weight": np.ones((4,), dtype="float32"),
            "norm.bias": np.zeros((4,), dtype="float32"),
            "position_ids": np.arange(4, dtype="int64"),
        }
        roles = {
            "dense.weight": ("parameter", "weight"),
            "dense.bias": ("parameter", "bias"),
            "norm.weight": ("parameter", "weight"),
            "norm.bias": ("parameter", "bias"),
            "position_ids": None,
        }

        def draw(key):
            shape = tensors[key].shape
            return reference_parity.unit_draw(0, key, shape)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.safetensors")
            save_file(tensors, path, metadata={"format": "pt"})
            reference_parity.redraw(path, roles)
            redrawn, metadata = reference_parity.read_safetensors(path)
            self.assertEqual(metadata, {"format": "pt"})
            self.assertAllClose(
                redrawn["dense.weight"], 0.6 * draw("dense.weight")
            )
            self.assertAllClose(
                redrawn["dense.bias"], 0.25 * draw("dense.bias")
            )
            self.assertAllClose(
                redrawn["norm.weight"], 1.0 + 0.25 * draw("norm.weight")
            )
            self.assertAllClose(redrawn["norm.bias"], 0.25 * draw("norm.bias"))
            self.assertAllEqual(redrawn["position_ids"], np.arange(4))

    def test_hidden_progress_bars(self):
        pytest.importorskip("transformers")
        from transformers.utils import logging

        hub_tqdm = importlib.import_module("huggingface_hub.utils.tqdm")
        if not isinstance(getattr(hub_tqdm, "progress_bar_states", None), dict):
            self.skipTest(
                "huggingface_hub keeps no `progress_bar_states`; "
                "test_hidden_progress_bars_without_hub_states covers this."
            )

        def settings():
            return (
                logging.is_progress_bar_enabled(),
                dict(hub_tqdm.progress_bar_states),
            )

        def restore(setting):
            enabled, hub_states = setting
            if enabled:
                logging.enable_progress_bar()
            else:
                logging.disable_progress_bar()
            hub_tqdm.progress_bar_states.clear()
            hub_tqdm.progress_bar_states.update(hub_states)

        before = settings()
        try:
            # The settings of huggingface_hub include those of its groups.
            for setting in (
                (True, {}),
                (True, {"_global": True, "huggingface_hub.lfs_upload": False}),
                (False, {"_global": False}),
                (False, {"_global": False, "huggingface_hub.lfs_upload": True}),
            ):
                restore(setting)
                with self.assertRaises(ValueError):
                    with reference_parity.hidden_progress_bars():
                        self.assertFalse(logging.is_progress_bar_enabled())
                        raise ValueError()
                self.assertEqual(settings(), setting)
        finally:
            restore(before)

    def test_hidden_progress_bars_without_hub_states(self):
        pytest.importorskip("transformers")
        from transformers.utils import logging

        hub_utils = importlib.import_module("huggingface_hub.utils")
        hub_tqdm = importlib.import_module("huggingface_hub.utils.tqdm")
        # A huggingface_hub without `progress_bar_states`, which is not
        # public API: its public functions keep the setting elsewhere.
        hub_disabled = [False]

        def set_hub(disabled):
            hub_disabled[0] = disabled

        def settings():
            return logging.is_progress_bar_enabled(), hub_disabled[0]

        def restore(setting):
            if setting[0]:
                logging.enable_progress_bar()
            else:
                logging.disable_progress_bar()
            set_hub(setting[1])

        hub_states = vars(hub_tqdm).pop("progress_bar_states", None)
        if hub_states is not None:
            self.addCleanup(
                setattr, hub_tqdm, "progress_bar_states", hub_states
            )
        patch = mock.patch.multiple(
            hub_utils,
            are_progress_bars_disabled=lambda: hub_disabled[0],
            disable_progress_bars=lambda: set_hub(True),
            enable_progress_bars=lambda: set_hub(False),
        )
        patch.start()
        self.addCleanup(patch.stop)
        # Cleanups run last first: this one inside the patch, so that the
        # real settings of huggingface_hub stay as they were.
        self.addCleanup(restore, settings())
        for setting in (
            (True, False),
            (True, True),
            (False, True),
            (False, False),
        ):
            restore(setting)
            with self.assertRaises(ValueError):
                with reference_parity.hidden_progress_bars():
                    self.assertFalse(logging.is_progress_bar_enabled())
                    raise ValueError()
            self.assertEqual(settings(), setting)

    def test_text_inputs(self):
        token_ids, padding_mask = reference_parity.text_inputs(211)
        self.assertEqual(token_ids.shape, (4, 17))
        self.assertTrue(np.all(token_ids >= 5) and np.all(token_ids < 211))
        self.assertAllEqual(token_ids, reference_parity.text_inputs(211)[0])
        valid = [list(np.flatnonzero(row)) for row in padding_mask]
        self.assertEqual(valid[0], list(range(17)))
        self.assertEqual(valid[1], list(range(11)))
        self.assertEqual(valid[2], list(range(6, 17)))
        self.assertEqual(valid[3], [0, 1, 2] + list(range(6, 14)))

    def test_keras_inputs(self):
        token_ids, padding_mask = reference_parity.text_inputs(211)
        backbone = mock.Mock()
        backbone.input = {"token_ids": None, "padding_mask": None}
        inputs = reference_parity.keras_inputs(
            backbone, token_ids, padding_mask
        )
        self.assertEqual(sorted(inputs), ["padding_mask", "token_ids"])
        backbone.input["segment_ids"] = None
        inputs = reference_parity.keras_inputs(
            backbone, token_ids, padding_mask
        )
        self.assertAllEqual(inputs["segment_ids"][0], [0] * 8 + [1] * 9)

    def test_per_position_error(self):
        reference = np.array([[[2.0, 2.0, 2.0, 2.0], [0.01, 0.01, 0.01, 0.01]]])
        delta = np.array([[[0.0, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.002]]])
        status, error = reference_parity.per_position_error(
            reference + delta, reference
        )
        self.assertEqual(status, "ok")
        # max(0.3 / 2, 0.002 / 0.01): each difference relative to the root
        # mean square of the reference at its position.
        self.assertAllClose(error, 0.2)

    def test_per_position_error_statuses(self):
        error = reference_parity.per_position_error
        reference = np.zeros((2, 3, 4))
        nan = reference.copy()
        nan[0, 1, 2] = np.nan
        inf = reference.copy()
        inf[1, 2, 3] = np.inf
        self.assertEqual(error(np.zeros((2, 3, 5)), reference), ("shape", None))
        self.assertEqual(error(nan, reference), ("nonfinite_keras", None))
        self.assertEqual(error(inf, reference), ("nonfinite_keras", None))
        self.assertEqual(error(reference, nan), ("nonfinite_reference", None))
        # Masked positions are left out, even when they are not finite.
        mask = np.ones((2, 3))
        mask[0, 1] = 0
        self.assertEqual(error(nan, reference, mask), ("ok", 0.0))
        # Where the reference is zero, any other output fails.
        self.assertGreater(error(reference + 1e-9, reference)[1], 1.0)

    def test_check_parity(self):
        check = reference_parity.check_parity
        self.assertIsNone(check({"hidden": ("ok", 1e-5)}))
        self.assertIn(
            "hidden 2.00e-04",
            check({"hidden": ("ok", 2e-4), "logits": ("ok", 0)}),
        )
        # A failing value never prints as the threshold itself.
        self.assertEqual(
            check({"hidden": ("ok", 1.04e-4)}),
            "per-position error above 1e-04: hidden 1.04e-04",
        )
        self.assertIn(
            "logits: NaN or inf in the Keras output",
            check({"hidden": ("ok", 0.0), "logits": ("nonfinite_keras", None)}),
        )

    def test_judge_checks(self):
        def judge(**errors):
            comparisons = {
                check: {"hidden": ("ok", error)}
                for check, error in errors.items()
            }
            return reference_parity.judge_checks(comparisons)

        # A failing check is a failure, with a hint.
        failures = judge(revealing=2e-4)
        self.assertLen(failures, 1)
        self.assertStartsWith(
            failures[0],
            "revealing: per-position error above 1e-04: hidden 2.00e-04\n  "
            "Weights drawn at full scale: look for a swapped, transposed",
        )
        # A check that did not run, as after a failed load, is not judged.
        self.assertEqual(judge(), [])

    def run_with_errors(self, errors, problems=()):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("safetensors")
        comparisons = {
            check: {"hidden": ("ok", error)} for check, error in errors.items()
        }
        outcome = (
            list(problems),
            comparisons,
            ["converter: convert_gpt2"],
        )
        with mock.patch.object(
            reference_parity, "run_checks", return_value=outcome
        ):
            self.run_reference_parity_test(
                model_type="gpt2",
                reference_class="GPT2LMHeadModel",
                config=GPT2_CONFIG,
                backbone_cls=GPT2Backbone,
            )

    def test_failures_fail_the_test(self):
        passing = {"revealing": 1e-5}
        failing = {"revealing": 1.3e-2}
        self.run_with_errors(passing)
        with self.assertRaisesRegex(
            AssertionError, "revealing: per-position error above 1e-04"
        ):
            self.run_with_errors(failing)
        # Problems fail the test, even when every parity check passes.
        with self.assertRaisesRegex(AssertionError, "negative control"):
            self.run_with_errors(passing, ["negative control: blind"])

    def test_describe_load_failure(self):
        pytest.importorskip("transformers")

        def convert_weights():
            raise KeyError("h.0.attn.c_attn.weight")

        try:
            convert_weights()
        except KeyError as e:
            description = reference_parity.describe_load_failure(
                "Backbone.from_preset", e
            )
        # The frame that raised is named, with its file and line.
        self.assertIn("raised KeyError: 'h.0.attn.c_attn.weight'", description)
        self.assertIn(f'File "{__file__}", line', description)
        self.assertIn(
            'in convert_weights\n    raise KeyError("h.0', description
        )

    def test_errors_of_the_load(self):
        for module in ("torch", "transformers", "safetensors"):
            pytest.importorskip(module)
        arguments = {
            "model_type": "gpt2",
            "reference_class": "GPT2LMHeadModel",
            "config": GPT2_CONFIG,
            "backbone_cls": GPT2Backbone,
            "task_cls": GPT2CausalLM,
        }

        def convert_weights(backbone, loader, transformers_config):
            raise KeyError("h.0.attn.c_attn.weight")

        # What the converter raises is a failed load, shown with its frame.
        with mock.patch.object(
            convert_gpt2, "convert_weights", convert_weights
        ):
            with self.assertRaisesRegex(
                AssertionError,
                r"load: `GPT2CausalLM.from_preset` raised KeyError: "
                r"'h.0.attn.c_attn.weight'\n(.*\n)*.*in convert_weights",
            ):
                self.run_reference_parity_test(**arguments)
