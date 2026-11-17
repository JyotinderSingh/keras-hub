import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.tests.test_case import TestCase

GEMMA2_CONFIG = {
    "vocab_size": 211,
    "hidden_size": 72,
    "intermediate_size": 100,
    "num_hidden_layers": 3,
    "num_attention_heads": 6,
    "num_key_value_heads": 2,
    "head_dim": 10,
    # Queries scaled by `head_dim`. `test_reference_parity_gemma2_query_scale`
    # tests the other scale.
    "query_pre_attn_scalar": 10,
    "sliding_window": 8,
    "max_position_embeddings": 64,
}
# Why the epsilon check of every Gemma and Gemma 2 test fails.
EPSILON_IGNORED = (
    "convert_gemma does not map `rms_norm_eps`, and GemmaBackbone passes "
    "`layer_norm_epsilon` to its final norm only, so the decoder blocks "
    "keep their 1e-6 default."
)


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = GemmaCausalLM.from_preset("hf://ariG23498/tiny-gemma-test")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

        model = GemmaCausalLM.from_preset("hf://ariG23498/tiny-gemma-2-test")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://ariG23498/tiny-gemma-test",
            load_weights=False,
        )
        self.assertIsInstance(model, GemmaCausalLM)
        model = Backbone.from_preset(
            "hf://ariG23498/tiny-gemma-test",
            load_weights=False,
        )
        self.assertIsInstance(model, GemmaBackbone)

    def test_reference_parity_gemma(self):
        self.run_reference_parity_test(
            model_type="gemma",
            reference_class="GemmaForCausalLM",
            config={
                "vocab_size": 211,
                "hidden_size": 72,
                "intermediate_size": 100,
                "num_hidden_layers": 3,
                "num_attention_heads": 6,
                "num_key_value_heads": 2,
                "head_dim": 10,
                "max_position_embeddings": 64,
            },
            backbone_cls=GemmaBackbone,
            task_cls=GemmaCausalLM,
            known_failures={"epsilon": (EPSILON_IGNORED, 58)},
        )

    def test_reference_parity_gemma2(self):
        self.run_reference_parity_test(
            model_type="gemma2",
            reference_class="Gemma2ForCausalLM",
            config=GEMMA2_CONFIG,
            backbone_cls=GemmaBackbone,
            task_cls=GemmaCausalLM,
            known_failures={"epsilon": (EPSILON_IGNORED, 39)},
        )

    def test_reference_parity_gemma2_query_scale(self):
        # Queries scaled by `hidden_size / num_attention_heads` (24), the
        # other scale keras-hub supports. It is far from `head_dim` (4), so
        # that every check fails clearly: with 12 against 10, low_variance
        # passed just under the threshold. A test of its own, because its
        # known failures would hide every smaller defect in the test above.
        query_scale = (
            "GemmaDecoderBlock always passes `query_head_dim_normalize=True`, "
            "so queries are scaled by `head_dim`, not by "
            "`query_pre_attn_scalar`."
        )
        self.run_reference_parity_test(
            model_type="gemma2",
            reference_class="Gemma2ForCausalLM",
            config=dict(
                GEMMA2_CONFIG,
                hidden_size=96,
                num_attention_heads=4,
                head_dim=4,
                query_pre_attn_scalar=24,
            ),
            backbone_cls=GemmaBackbone,
            task_cls=GemmaCausalLM,
            known_failures={
                "revealing": (query_scale, 4.1),
                "low_variance": (query_scale, 7.3e-3),
                "epsilon": (EPSILON_IGNORED, 50),
            },
        )
