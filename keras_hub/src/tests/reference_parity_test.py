import contextlib
import math
from unittest import mock

import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.models.modern_bert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modern_bert.modern_bert_layers import (
    ModernBertEncoderLayer,
)
from keras_hub.src.models.modern_bert.modern_bert_masked_lm import (
    ModernBertMaskedLM,
)
from keras_hub.src.models.qwen.qwen_causal_lm import QwenCausalLM
from keras_hub.src.tests import reference_parity
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_qwen
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

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
QWEN2_CONFIG = {
    "vocab_size": 211,
    "hidden_size": 72,
    "intermediate_size": 100,
    "num_hidden_layers": 2,
    "num_attention_heads": 6,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
}
MODERN_BERT_CONFIG = {
    "vocab_size": 211,
    "hidden_size": 48,
    "intermediate_size": 80,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "max_position_embeddings": 64,
    "local_attention": 8,
    "global_attn_every_n_layers": 2,
    "pad_token_id": 0,
    "cls_token_id": 1,
    "sep_token_id": 2,
    "bos_token_id": 1,
    "eos_token_id": 2,
}
GPT2 = {
    "model_type": "gpt2",
    "reference_class": "GPT2LMHeadModel",
    "cls": GPT2CausalLM,
    "config": GPT2_CONFIG,
}
QWEN2 = {
    "model_type": "qwen2",
    "reference_class": "Qwen2ForCausalLM",
    "cls": QwenCausalLM,
    "config": QWEN2_CONFIG,
}
MODERN_BERT = {
    "model_type": "modernbert",
    "reference_class": "ModernBertModel",
    "cls": ModernBertBackbone,
    "config": MODERN_BERT_CONFIG,
}
MISTRAL = {"model_type": "mistral", "reference_class": "MistralForCausalLM"}
REASON = "The converter rounds every weight."
THRESHOLD = reference_parity.THRESHOLD


def seed(defect):
    """Returns a patch that puts `defect` in a converter or a model."""
    if defect is None:
        return contextlib.nullcontext()
    if defect == "gate_and_up_swapped":
        port_weight = SafetensorLoader.port_weight
        swap = {"gate_proj": "up_proj", "up_proj": "gate_proj"}

        # Converters pass `port_weight` its arguments by keyword.
        def port_swapped(loader, keras_variable, hf_weight_key, hook_fn=None):
            parts = hf_weight_key.split(".")
            key = ".".join(swap.get(part, part) for part in parts)
            port_weight(loader, keras_variable, key, hook_fn)

        return mock.patch.object(SafetensorLoader, "port_weight", port_swapped)
    if defect == "mlp_norm_epsilon_hard_coded":
        init = ModernBertEncoderLayer.__init__

        def init_seeded(layer, *args, **kwargs):
            init(layer, *args, **kwargs)
            layer.mlp_norm.epsilon = 1e-5

        return mock.patch.object(
            ModernBertEncoderLayer, "__init__", init_seeded
        )
    # "epsilon_dropped" or "epsilon_hard_coded", in convert_qwen.
    convert = convert_qwen.convert_backbone_config

    def convert_seeded(transformers_config):
        config = convert(transformers_config)
        del config["layer_norm_epsilon"]
        if defect == "epsilon_hard_coded":
            config["layer_norm_epsilon"] = 1e-5
        return config

    return mock.patch.object(
        convert_qwen, "convert_backbone_config", convert_seeded
    )


class ReferenceParityTest(TestCase):
    def test_per_position_error(self):
        error = reference_parity.per_position_error
        reference = np.array([[[2.0, 2.0, 2.0, 2.0], [0.01] * 4]])
        output = reference + [[[0.0, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.002]]]
        # Each difference is relative to the root mean square of the
        # reference at its position: max(0.3 / 2, 0.002 / 0.01).
        self.assertAllClose(error(output, reference)[0], 0.2)
        # Masked positions are ignored, even when they are not finite.
        output[0, 1, 0] = np.nan
        self.assertAllClose(error(output, reference, [[1, 0]])[0], 0.15)
        infinite = reference.copy()
        infinite[0, 0, 0] = np.inf
        for keras_output, reference_output, note in (
            (np.zeros((1, 2, 5)), reference, "shapes differ"),
            (output, reference, "NaN or inf in the Keras output"),
            (reference, infinite, "NaN or inf in the transformers output"),
        ):
            value, text = error(keras_output, reference_output)
            self.assertEqual(value, math.inf)
            self.assertIn(note, text)

    def test_draws_and_inputs(self):
        # Every parity test depends on these values; change them on purpose.
        draw = reference_parity.unit_draw(0, "wte.weight", (2, 3))
        self.assertAllEqual(
            draw,
            [
                [-1.1977031545881198, 0.8892476811345476, -0.7633708311511904],
                [0.6460901334155383, -0.3194623459182225, 0.1709566494281259],
            ],
        )
        other = reference_parity.unit_draw(0, "wpe.weight", (2, 3))
        self.assertFalse(np.array_equal(draw, other))

        backbone = mock.Mock(input={"token_ids": None, "padding_mask": None})
        inputs = reference_parity.keras_inputs(backbone, 211)
        self.assertEqual(sorted(inputs), ["padding_mask", "token_ids"])
        token_ids = inputs["token_ids"]
        self.assertEqual(token_ids.shape, (4, 17))
        self.assertTrue(np.all((token_ids >= 5) & (token_ids < 211)))
        self.assertAllEqual(token_ids[0, :4], [180, 136, 110, 60])
        # All valid, right-padded, left-padded, and a hole with right
        # padding.
        mask = ["".join(map(str, row)) for row in inputs["padding_mask"]]
        self.assertEqual(
            mask,
            [
                "11111111111111111",
                "11111111111000000",
                "00000011111111111",
                "11100011111111000",
            ],
        )
        backbone.input["segment_ids"] = None
        inputs = reference_parity.keras_inputs(backbone, 211)
        self.assertAllEqual(inputs["segment_ids"], [[0] * 8 + [1] * 9] * 4)

    def test_redraw(self):
        torch = pytest.importorskip("torch")
        for scale in (1.0, 0.01):
            model = torch.nn.Module()
            model.dense = torch.nn.Linear(16, 9)
            model.norm = torch.nn.LayerNorm(9)
            reference_parity.redraw(model, scale)
            # Each parameter: (center, standard deviation of the noise).
            expected = {
                # 1.2 / sqrt(9): its smaller dimension.
                "dense.weight": (0.0, 0.4 * scale),
                # The constructed value if constant, else 0: Linear draws
                # its bias.
                "dense.bias": (0.0, 0.25 * scale),
                # Normalization layers keep full noise.
                "norm.weight": (1.0, 0.25),
                "norm.bias": (0.0, 0.25),
            }
            for name, value in model.state_dict().items():
                center, std = expected[name]
                noise = reference_parity.unit_draw(0, name, tuple(value.shape))
                self.assertAllClose(value.numpy(), center + std * noise)

    @parameterized.named_parameters(
        (
            "unclassified_field",
            {"config": dict(GPT2_CONFIG, hidden_act="relu")},
            "config: `hidden_act`",
        ),
        (
            "epsilon_field",
            {"config": dict(GPT2_CONFIG, layer_norm_epsilon=1e-6)},
            "config: `layer_norm_epsilon`",
        ),
        (
            "default_value",
            {"config": dict(GPT2_CONFIG, n_head=12)},
            "config: `n_head=12`",
        ),
        (
            "out_of_bounds",
            {"config": dict(GPT2_CONFIG, n_layer=1)},
            "config: `n_layer`",
        ),
        (
            "not_an_int",
            {"config": dict(GPT2_CONFIG, n_layer=3.0)},
            "config: `n_layer` must be an int",
        ),
        (
            # A token id may keep its default (MistralConfig's
            # bos_token_id is 1): the window is the only problem.
            "token_id_default",
            dict(MISTRAL, config=dict(QWEN2_CONFIG, bos_token_id=1)),
            "config: `sliding_window` defaults to",
        ),
        (
            # A 0 the test writes is checked; only a default 0 means off.
            "written_zero",
            dict(QWEN2, config=dict(QWEN2_CONFIG, num_hidden_layers=0)),
            "config: `num_hidden_layers` is 0",
        ),
        (
            # MistralConfig defaults to a window of 4096.
            "default_out_of_bounds",
            dict(MISTRAL, config=QWEN2_CONFIG),
            "config: `sliding_window` defaults to",
        ),
        (
            "no_grouped_query_attention",
            dict(QWEN2, config=dict(QWEN2_CONFIG, num_key_value_heads=6)),
            "config: `num_key_value_heads`",
        ),
        (
            "computed_head_dim",
            dict(QWEN2, config=dict(QWEN2_CONFIG, head_dim=12)),
            "config: `head_dim=12`",
        ),
        (
            "model_type_mismatch",
            {"model_type": "qwen2"},
            "model_type: `GPT2LMHeadModel`",
        ),
        (
            "not_a_transformers_class",
            {"reference_class": "NotAModel"},
            "reference_class: `NotAModel`",
        ),
        (
            "known_failure_not_a_check",
            {"known_failures": {"unread": (REASON, 1.0)}},
            "known_failures: `unread`",
        ),
        (
            "recorded_at_threshold",
            {"known_failures": {"epsilon": (REASON, THRESHOLD)}},
            "known_failures: `epsilon`",
        ),
        (
            "recorded_inf",
            {"known_failures": {"epsilon": (REASON, math.inf)}},
            "known_failures: `epsilon`",
        ),
        (
            "known_failure_without_reason",
            {"known_failures": {"epsilon": ("", 1.0)}},
            "known_failures: `epsilon` needs",
        ),
        (
            "exemption_without_reason",
            {"ignored_reference_tensors": {"lm_head.*": ""}},
            "ignored_reference_tensors: `lm_head.*`",
        ),
        (
            # Only causal LMs among tasks: other tasks take other inputs.
            "unsupported_cls",
            {"cls": ModernBertMaskedLM},
            "cls: ",
        ),
    )
    def test_bad_arguments_fail_before_building(self, arguments, problem):
        built = RuntimeError("A model was built.")
        with mock.patch.object(
            reference_parity, "run_checks", side_effect=built
        ):
            with self.assertRaises(AssertionError) as context:
                self.run_reference_parity_test(**dict(GPT2, **arguments))
        # The header, then this problem only.
        lines = str(context.exception).splitlines()
        self.assertLen(lines, 2)
        self.assertStartsWith(lines[1], problem)

    def run_with_errors(self, errors, problems=(), known=None):
        outcome = (errors, list(problems), ["converter: convert_gpt2"])
        with mock.patch.object(
            reference_parity, "run_checks", return_value=outcome
        ):
            self.run_reference_parity_test(**GPT2, known_failures=known)

    def test_known_failures_are_strict(self):
        passing = {"full_weights": 1e-5, "small_weights": 1e-6, "epsilon": 1e-6}
        failing = dict(passing, small_weights=1.3e-2)
        known = {"small_weights": (REASON, 1e-2)}
        self.run_with_errors(passing)
        # Within the band, the test shows as XFAIL, with the reason.
        # `pytest.xfail` is patched, so that this test passes under
        # `pytest --runxfail` too.
        xfailed = pytest.xfail.Exception
        with mock.patch.object(pytest, "xfail", side_effect=xfailed) as xfail:
            with self.assertRaises(xfailed):
                self.run_with_errors(failing, known=known)
        message = xfail.call_args.args[0]
        self.assertIn("\nKnown failures:\n  small_weights: 0.013 now", message)
        self.assertIn(REASON, message)
        # `pytest --runxfail` makes `pytest.xfail` return: the test fails.
        with mock.patch.object(pytest, "xfail") as xfail:
            with self.assertRaisesRegex(AssertionError, "Known failures:"):
                self.run_with_errors(failing, known=known)
        xfail.assert_called_once()

        # Anything else fails, and never shows as XFAIL.
        grown = dict(passing, small_weights=0.05)
        shrunk = dict(passing, small_weights=4e-3)
        infinite = dict(passing, small_weights=math.inf)
        not_run = {"full_weights": 1e-5, "small_weights": 1e-6}
        with mock.patch.object(pytest, "xfail") as xfail:
            for errors, known_failures, expected in (
                (failing, None, "\nsmall_weights: 0.013, above"),
                # A known failure excuses only its own check.
                (dict(failing, full_weights=1.0), known, "\nfull_weights: 1,"),
                (passing, known, "\nknown failure small_weights: no longer"),
                (grown, known, "do not raise"),
                (shrunk, known, "record 0.004"),
                (infinite, known, "\nknown failure small_weights: inf"),
                (not_run, {"epsilon": (REASON, 1.0)}, "did not run"),
            ):
                with self.assertRaises(AssertionError) as context:
                    self.run_with_errors(errors, known=known_failures)
                self.assertIn(expected, str(context.exception))
            # Problems are never excused.
            with self.assertRaisesRegex(AssertionError, "\nunread:"):
                self.run_with_errors(failing, ["unread: lm_head.bias"], known)
        xfail.assert_not_called()

    def test_routing_and_logits(self):
        for arguments, problem in (
            # A base class: `Backbone.from_preset` returns GPT2Backbone.
            ({"cls": Backbone}, "load: returned GPT2Backbone, not Backbone"),
            # GPT2Model has no head: the logits of the task go unchecked.
            ({"reference_class": "GPT2Model"}, "load: `GPT2Model` has no"),
        ):
            with self.assertRaises(AssertionError) as context:
                self.run_reference_parity_test(**dict(GPT2, **arguments))
            self.assertIn("\n" + problem, str(context.exception))

    def test_audit(self):
        problems = reference_parity.check_patterns(
            "ignored_reference_tensors",
            ["pooler.*", "*.weight"],
            ["h.0.weight", "lm_head.weight"],
            ["lm_head.weight"],
            "reads",
        )
        self.assertLen(problems, 2)
        self.assertIn("`pooler.*` matches nothing", problems[0])
        self.assertIn("`*.weight` also matches `h.0.weight`", problems[1])

        port_weight = SafetensorLoader.port_weight

        def skip_norm(loader, keras_variable, hf_weight_key, hook_fn=None):
            if hf_weight_key not in ("ln_f.weight", "ln_f.bias"):
                port_weight(loader, keras_variable, hf_weight_key, hook_fn)

        with mock.patch.object(SafetensorLoader, "port_weight", skip_norm):
            with self.assertRaises(AssertionError) as context:
                self.run_reference_parity_test(
                    **GPT2,
                    ignored_reference_tensors={
                        "transformer.ln_f.weight": "Seeded."
                    },
                    derived_weights={"layer_norm/gamma": "Seeded."},
                )
        # Each label lists only what no pattern excuses.
        message = str(context.exception)
        self.assertRegex(
            message, r"\nunread: 1 of .*\n    transformer\.ln_f\.bias\n  Port"
        )
        self.assertRegex(
            message, r"\nunassigned: 1 of .*\n    layer_norm/beta .*\n  Port"
        )

    @parameterized.named_parameters(
        ("no_defect", QWEN2, None, set()),
        # small_weights measures 5.2e-5: with small weights, the swap
        # changes the MLP only at third order.
        ("gate_and_up_swapped", QWEN2, "gate_and_up_swapped", {"full_weights"}),
        ("epsilon_dropped", QWEN2, "epsilon_dropped", {"epsilon"}),
        (
            "epsilon_hard_coded",
            QWEN2,
            "epsilon_hard_coded",
            {"full_weights", "small_weights", "epsilon"},
        ),
        # The weakest case known: the check measures 2.5e-4.
        (
            "mlp_norm_epsilon_hard_coded",
            MODERN_BERT,
            "mlp_norm_epsilon_hard_coded",
            {"epsilon"},
        ),
    )
    def test_seeded_defects(self, arguments, defect, expected):
        message = ""
        with seed(defect):
            try:
                self.run_reference_parity_test(**arguments)
            except AssertionError as error:
                message = str(error)
        # The labels of the lines that fail the test.
        labels = {
            line.split(":")[0]
            for line in message.splitlines()[1:]
            if not line.startswith(" ")
        }
        self.assertEqual(labels - {"Measured"}, expected, message)
