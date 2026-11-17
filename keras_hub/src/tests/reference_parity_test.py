import importlib
import os
import tempfile
import traceback
from unittest import mock

import numpy as np
import pytest
from keras.src.backend.common.variables import Variable

from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.tests import reference_parity
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_gpt2
from keras_hub.src.utils.transformers import preset_loader
from keras_hub.src.utils.transformers import safetensor_utils

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
# The reasons of known failures in these tests.
MISTRAL_FLOAT16 = (
    "convert_mistral rounds every weight to float16, so a float32 "
    "checkpoint does not load exactly."
)


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

    def test_check_config_accepts(self):
        pytest.importorskip("transformers")
        check = reference_parity.check_config
        self.assertEqual(check("GPT2LMHeadModel", GPT2_CONFIG), [])
        # Equal to the default, but exempt: convert_bert supports only 2.
        config = {"vocab_size": 211, "type_vocab_size": 2, "pad_token_id": 0}
        self.assertEqual(check("BertModel", config), [])

    def test_check_config_rejects_unclassified_fields(self):
        pytest.importorskip("transformers")
        for field, value in (
            ("hidden_act", "gelu"),
            ("attention_dropout", 0.1),
            ("attention_bias", True),
            ("rope_theta", 100.0),
            ("not_a_field", 1),
        ):
            self.assertEqual(
                reference_parity.check_config(
                    "LlamaForCausalLM", {field: value}
                ),
                [
                    f"`{field}` is not a field a parity test may set; ask a "
                    f"maintainer to classify `{field}`."
                ],
            )

    def test_check_config_rejects_bad_values(self):
        pytest.importorskip("transformers")
        for field, value, message in (
            ("vocab_size", 16, "`vocab_size=16` is out of bounds"),
            ("vocab_size", 1000, "`vocab_size=1000` is out of bounds"),
            ("n_embd", 128, "`n_embd=128` is out of bounds"),
            ("n_layer", 1, "`n_layer=1` is out of bounds"),
            ("n_layer", 7, "`n_layer=7` is out of bounds"),
            ("n_head", 1, "`n_head=1` is out of bounds"),
            ("n_positions", 16, "`n_positions=16` is out of bounds"),
            ("n_positions", 256, "`n_positions=256` is out of bounds"),
            ("bos_token_id", 5, "`bos_token_id=5` is out of bounds"),
            # A window as long as the inputs would leave it untested.
            ("sliding_window", 4096, "`sliding_window=4096` is out of bounds"),
            ("sliding_window", 17, "`sliding_window=17` is out of bounds"),
            ("n_layer", 3.0, "`n_layer` must be an integer"),
            ("n_layer", True, "`n_layer` must be an integer"),
        ):
            config = dict(GPT2_CONFIG, **{field: value})
            problems = reference_parity.check_config("GPT2LMHeadModel", config)
            self.assertLen(problems, 1)
            self.assertIn(message, problems[0])

    def test_check_config_rejects_head_dim(self):
        pytest.importorskip("transformers")
        config = {
            "hidden_size": 72,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,
            "head_dim": 10,
        }
        check = reference_parity.check_config
        self.assertEqual(check("LlamaForCausalLM", config), [])
        # A converter that computes the head size would pass with 12.
        self.assertEqual(
            check("LlamaForCausalLM", dict(config, head_dim=12)),
            [
                "`head_dim=12` equals `hidden_size // num_attention_heads`; "
                "use another value so that the test shows the converter "
                "reads `head_dim`."
            ],
        )
        self.assertEqual(
            check("LlamaForCausalLM", dict(config, head_dim=1)),
            ["`head_dim=1` is out of bounds: a size field must be at least 4."],
        )

    def test_check_config_rejects_derived_values(self):
        pytest.importorskip("transformers")
        check = reference_parity.check_config
        config = {"hidden_size": 72, "num_attention_heads": 6}
        kv_heads = (
            "`num_key_value_heads` is {}; set it to more than 1 and less than "
            "`num_attention_heads` (6), so that the test runs grouped-query "
            "attention."
        )
        # LlamaConfig sets `num_key_value_heads` to `num_attention_heads`
        # when it is not given.
        self.assertEqual(
            check("LlamaForCausalLM", config), [kv_heads.format(6)]
        )
        self.assertEqual(
            check("LlamaForCausalLM", dict(config, num_key_value_heads=6)),
            [kv_heads.format(6)],
        )
        self.assertEqual(
            check("LlamaForCausalLM", dict(config, num_key_value_heads=2)), []
        )
        # One key/value head is below the bound of a heads field.
        self.assertEqual(
            check("LlamaForCausalLM", dict(config, num_key_value_heads=1)),
            [
                "`num_key_value_heads=1` is out of bounds: a heads field must "
                "be at least 2."
            ],
        )
        # A value that the config class derives must be within bounds too:
        # the head size here, and below the window of ModernBERT, whose
        # `local_attention` is twice `sliding_window`.
        self.assertEqual(
            check(
                "LlamaForCausalLM",
                {
                    "hidden_size": 16,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                },
            ),
            [
                "`head_dim=2` is out of bounds: a size field must be at least "
                "4. LlamaConfig derives it from the fields that the test sets."
            ],
        )
        self.assertEqual(
            check("ModernBertModel", {"sliding_window": 12}),
            [
                "`local_attention=24` is out of bounds: a window field must "
                "be 2 to 16. ModernBertConfig derives it from the fields that "
                "the test sets."
            ],
        )

    def test_check_config_rejects_default_shape(self):
        pytest.importorskip("transformers")
        problems = reference_parity.check_config(
            "LlamaForCausalLM", {"intermediate_size": 11008}
        )
        self.assertEqual(
            problems,
            [
                "`intermediate_size=11008` equals the LlamaConfig default; use "
                "another value so that the test shows the converter reads "
                "`intermediate_size`."
            ],
        )
        self.assertEqual(
            reference_parity.check_config("NotAModel", {}),
            ["`NotAModel` is not a transformers class."],
        )

    def test_check_config_rejects_default_structure(self):
        pytest.importorskip("transformers")
        check = reference_parity.check_config
        # A converter that ignores the top-k of Mixtral would pass with the
        # default of 2, as the Keras backbone defaults to 2 too.
        self.assertEqual(
            check("MixtralForCausalLM", {"num_experts_per_tok": 2}),
            [
                "`num_experts_per_tok=2` equals the MixtralConfig default; "
                "use another value so that the test shows the converter "
                "reads `num_experts_per_tok`."
            ],
        )
        # A field in DEFAULT_FIELDS may keep its default.
        config = {
            "hidden_size": 72,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,
            "decoder_sparse_step": 1,
        }
        self.assertEqual(check("Qwen2MoeForCausalLM", config), [])

    def test_reference_parameter_cap(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        config = {
            "vocab_size": 512,
            "hidden_size": 96,
            "intermediate_size": 5000,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        }
        self.assertEqual(
            reference_parity.check_config("LlamaForCausalLM", config), []
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "more than 5,000,000"):
                reference_parity.build_reference(
                    "LlamaForCausalLM", config, directory
                )

    def test_check_reasons(self):
        check = reference_parity.check_reasons
        self.assertEqual(check("derived_weights", {"*inv_freq": "RoPE"}), [])
        self.assertEqual(
            check("derived_weights", {"*inv_freq": None, "*bias": " "}),
            [
                "`derived_weights`: `*inv_freq` needs a non-empty reason.",
                "`derived_weights`: `*bias` needs a non-empty reason.",
            ],
        )

    def test_check_known_failures(self):
        check = reference_parity.check_known_failures
        known = {"revealing": (MISTRAL_FLOAT16, 3e-4)}
        self.assertEqual(check(known), [])
        problems = check(
            {
                "unread": ("Not read.", 1.0),
                "revealing": MISTRAL_FLOAT16,
            }
        )
        self.assertLen(problems, 2)
        self.assertIn("`unread` is not a parity check", problems[0])
        for problem in problems[1:]:
            self.assertIn("needs a tuple `(reason, max_error)`", problem)
        # The reason must not be empty.
        self.assertLen(check({"revealing": (" ", 1e-2)}), 1)
        # `max_error` must be above the threshold.
        self.assertLen(check({"revealing": (MISTRAL_FLOAT16, 1e-4)}), 1)

    def test_check_patterns(self):
        names = ["h.0.weight", "h.0.bias", "lm_head.weight", "lm_head.bias"]
        unread = ["lm_head.weight", "lm_head.bias"]
        nouns = ("unread tensor", "read")
        check = reference_parity.check_patterns
        self.assertEqual(
            check("ignored", ["lm_head.*"], names, unread, nouns), []
        )
        self.assertEqual(
            check(
                "ignored", ["pooler.*", "h.0.*", "*[!s]"], names, unread, nouns
            ),
            [
                "`ignored`: `pooler.*` matches no unread tensor; remove it.",
                "`ignored`: `h.0.*` matches no unread tensor; remove it.",
                "`ignored`: `*[!s]` also matches `h.0.weight`, which the "
                "converter read; make it specific.",
            ],
        )

    def test_apply_known_failures(self):
        def apply(known_failures, **errors):
            comparisons = {
                check: {"hidden": ("ok", error)}
                for check, error in errors.items()
            }
            return reference_parity.apply_known_failures(
                comparisons, known_failures
            )

        known = {"revealing": (MISTRAL_FLOAT16, 3e-2)}
        # A known failure within its max_error is expected.
        self.assertEqual(
            apply(known, revealing=1.3e-2),
            (
                [],
                [
                    f"revealing: {MISTRAL_FLOAT16}\n  measured: per-position "
                    "error above 1e-04: hidden 1.30e-02 (max_error 0.03)"
                ],
            ),
        )
        # A failure that is not listed is a failure, with a hint.
        failures, expected = apply({}, revealing=2e-4)
        self.assertEqual(expected, [])
        self.assertLen(failures, 1)
        self.assertStartsWith(
            failures[0],
            "revealing: per-position error above 1e-04: hidden 2.00e-04\n  "
            "Weights drawn at full scale: look for a swapped, transposed",
        )
        # A known failure that no longer happens is a failure.
        self.assertEqual(
            apply(known, revealing=1e-5),
            (
                [
                    "known failure `revealing` no longer happens; remove it "
                    f"from known_failures (reason was: {MISTRAL_FLOAT16})"
                ],
                [],
            ),
        )
        # So is one above its max_error: a new defect, or the known one grew.
        failures, _ = apply(known, revealing=0.5)
        self.assertIn("This is above the max_error 0.03", failures[0])
        self.assertIn("look for a swapped, transposed", failures[0])
        failures, _ = reference_parity.apply_known_failures(
            {"revealing": {"hidden": ("nonfinite_keras", None)}}, known
        )
        self.assertIn("NaN or inf in the Keras output", failures[0])
        # And one far below it, whose max_error would hide a new defect.
        failures, _ = apply(known, revealing=1e-3)
        self.assertIn("less than 1/10 of its max_error 0.03", failures[0])
        self.assertIn("the remaining error is another defect", failures[0])
        # A known failure whose check did not run, as after a failed load,
        # is not judged.
        self.assertEqual(apply(known), ([], []))

    def run_with_errors(self, errors, problems=(), known_failures=None):
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
                known_failures=known_failures,
            )

    def test_known_failures_are_strict(self):
        passing = {"revealing": 1e-5}
        failing = {"revealing": 1.3e-2}
        known = {"revealing": (MISTRAL_FLOAT16, 3e-2)}
        self.run_with_errors(passing)
        # The test shows as XFAIL, with the reasons. `pytest.xfail` is
        # patched, so that this test passes under `pytest --runxfail` too.
        xfailed = pytest.xfail.Exception
        with mock.patch.object(pytest, "xfail", side_effect=xfailed) as xfail:
            with self.assertRaises(xfailed):
                self.run_with_errors(failing, known_failures=known)
        self.assertIn(MISTRAL_FLOAT16, xfail.call_args.args[0])
        # `pytest --runxfail` makes `pytest.xfail` return; the test fails.
        with mock.patch.object(pytest, "xfail") as xfail:
            with self.assertRaisesRegex(AssertionError, "convert_mistral"):
                self.run_with_errors(failing, known_failures=known)
        xfail.assert_called_once()
        with self.assertRaisesRegex(
            AssertionError, "revealing: per-position error above 1e-04"
        ):
            self.run_with_errors(failing)
        with self.assertRaisesRegex(
            AssertionError,
            "known failure `revealing` no longer happens; remove it",
        ):
            self.run_with_errors(passing, known_failures=known)
        # A new defect cannot hide behind a known failure.
        with self.assertRaisesRegex(AssertionError, "above the max_error"):
            self.run_with_errors({"revealing": 0.5}, known_failures=known)
        # Problems are never known failures.
        with self.assertRaisesRegex(AssertionError, "negative control"):
            self.run_with_errors(
                failing, ["negative control: blind"], known_failures=known
            )

    def test_bad_arguments_fail_before_running(self):
        pytest.importorskip("transformers")
        with mock.patch.object(reference_parity, "run_checks") as run_checks:
            with self.assertRaisesRegex(AssertionError, "layer_norm_epsilon"):
                self.run_reference_parity_test(
                    model_type="gpt2",
                    reference_class="GPT2LMHeadModel",
                    config=dict(GPT2_CONFIG, layer_norm_epsilon=1e-6),
                    backbone_cls=GPT2Backbone,
                    known_failures={"revealing": ""},
                )
        run_checks.assert_not_called()

    def test_check_audit(self):
        stored = [
            ("model.safetensors", key)
            for key in ("a.weight", "b.weight", "head.weight")
        ]
        audit = mock.Mock(
            reads={("model.safetensors", "a.weight")},
            failed_reads=[("model.safetensors", "a.gamma")],
            converter="convert_x",
        )
        weights = [
            mock.Mock(path=path, shape=(2,))
            for path in ("a/kernel", "b/kernel", "rope/inv_freq")
        ]
        census = {
            "never_assigned": weights[1:],
            "outside_phases": [],
            "nan_left": [],
        }
        problems = reference_parity.check_audit(
            audit,
            census,
            stored,
            weights,
            {"head.*": "A head."},
            {"rope/*": "Computed."},
        )
        # A failed read, such as a probe for another name, is no problem.
        self.assertEqual(
            problems,
            [
                "unread: 1 of 3 stored tensors were never read by "
                "`convert_x`:\n    b.weight\n  Port them, or list them in "
                "`ignored_reference_tensors` with the reason.",
                "unassigned: 1 of 3 Keras weights were never assigned by the "
                "converter:\n    b/kernel (2,)\n  Port them. A weight that "
                "the model computes from its config instead of loading goes "
                "in `derived_weights`, with the reason.",
            ],
        )

    def test_describe_weights(self):
        weights = [
            mock.Mock(path="rotary_emb/inv_freq", shape=(6,)),
            mock.Mock(path="dense/kernel", shape=(4, 4)),
            mock.Mock(path="rotary_emb/inv_freq", shape=(6,)),
        ]
        # A repeated path gets the index of the weight.
        self.assertEqual(
            reference_parity.describe_weights(weights[1:], weights),
            ["dense/kernel (4, 4)", "rotary_emb/inv_freq (6,) (weights[2])"],
        )

    def test_describe_load_failure(self):
        pytest.importorskip("transformers")
        note = "NaN in the message is the audit's placeholder"
        for message, has_note in (
            ("Got nan in the weights", True),
            ("NaN values", True),
            ("Down for maintenance", False),
        ):
            description = reference_parity.describe_load_failure(
                "Backbone.from_preset", ValueError(message)
            )
            self.assertEqual(note in description, has_note, message)

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

    def test_check_uninstrumented_names_suspects(self):
        error = RuntimeError("A weight has no value.")
        with mock.patch.object(
            reference_parity, "load_model", side_effect=error
        ):
            problems = reference_parity.check_uninstrumented(
                "unused",
                None,
                outputs={},
                inputs={},
                suspects=["inv_freq (6,)"],
            )
        self.assertLen(problems, 1)
        self.assertIn("RuntimeError: A weight has no value.", problems[0])
        self.assertIn("so look at them first:\n    inv_freq (6,)", problems[0])

    def test_load_audit_restores_patches(self):
        def patched():
            return (
                safetensor_utils.safetensors,
                preset_loader.TransformersPresetLoader.__init__,
            )

        originals = patched()
        assign = Variable.assign
        with self.assertRaises(ValueError):
            with reference_parity.LoadAudit():
                for value, original in zip(patched(), originals):
                    self.assertIsNot(value, original)
                raise ValueError()
        self.assertEqual(patched(), originals)
        self.assertIs(Variable.assign, assign)
        # A patch that fails, as after a rename in keras-hub, undoes those
        # made before it.
        with mock.patch.object(preset_loader, "TransformersPresetLoader", None):
            with self.assertRaises(AttributeError):
                with reference_parity.LoadAudit():
                    pass
        self.assertEqual(patched(), originals)

    def test_errors_of_the_load_and_of_the_audit(self):
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

        # A bug in the audit is not; it fails with its own traceback, also
        # when it raises inside `from_preset`.
        def snapshot(audit, weights):
            raise RuntimeError("bug in LoadAudit._snapshot")

        with mock.patch.object(
            reference_parity.LoadAudit, "_snapshot", snapshot
        ):
            try:
                self.run_reference_parity_test(**arguments)
            except RuntimeError as e:
                frames = traceback.extract_tb(e.__traceback__)
            else:
                self.fail("The bug in the audit was not raised.")
        # The traceback goes down to the frame that raised.
        self.assertEqual(frames[-1].name, "snapshot")
        for method in ("phase", "finish"):
            error = RuntimeError(f"bug in LoadAudit.{method}")
            with mock.patch.object(
                reference_parity.LoadAudit, method, side_effect=error
            ):
                with self.assertRaisesRegex(RuntimeError, "bug in LoadAudit"):
                    self.run_reference_parity_test(**arguments)
