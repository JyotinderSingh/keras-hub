import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM
from keras_hub.src.tests.test_case import TestCase

GEMMA3_TEXT_CONFIG = {
    "vocab_size": 211,
    "hidden_size": 72,
    "intermediate_size": 100,
    # Every sixth layer is a global attention layer, with its own RoPE base,
    # so six layers cover one.
    "num_hidden_layers": 6,
    "num_attention_heads": 6,
    "num_key_value_heads": 2,
    "head_dim": 10,
    # Queries scaled by `head_dim`. `test_reference_parity_query_scale` tests
    # the other scale.
    "query_pre_attn_scalar": 10,
    "sliding_window": 8,
    "max_position_embeddings": 64,
}
# Why the epsilon check of both Gemma 3 tests fails.
EPSILON_IGNORED = (
    "Gemma3Backbone passes `layer_norm_epsilon` to its final norm only, "
    "and Gemma3DecoderBlock does not pass it to its attention, so the "
    "block norms and the query and key norms keep their 1e-6 default."
)


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = Gemma3CausalLM.from_preset("google/gemma-3-1b-it")
        prompt = "What is the capital of India?"
        model.generate([prompt], max_length=15)

    @pytest.mark.extra_large
    def test_backbone_from_hf_preset(self):
        model = Gemma3Backbone.from_preset(
            "hf://microsoft/harrier-oss-v1-270m",
            load_weights=False,
        )
        # harrier 270m: hidden_dim=640, num_layers=18
        self.assertEqual(model.hidden_dim, 640)
        self.assertEqual(model.num_layers, 18)

    @pytest.mark.extra_large
    def test_class_detection(self):
        preset_name = "google/gemma-3-1b-it"
        model = CausalLM.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, Gemma3CausalLM)
        model = Backbone.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, Gemma3Backbone)

    def test_reference_parity(self):
        self.run_reference_parity_test(
            model_type="gemma3_text",
            reference_class="Gemma3ForCausalLM",
            config=GEMMA3_TEXT_CONFIG,
            backbone_cls=Gemma3Backbone,
            task_cls=Gemma3CausalLM,
            known_failures={"epsilon": (EPSILON_IGNORED, 44)},
        )

    def test_reference_parity_query_scale(self):
        # Queries scaled by `hidden_size / num_attention_heads`, the other
        # scale keras-hub supports. A test of its own, because its known
        # failures would hide every smaller defect in the test above.
        query_scale = (
            "Gemma3DecoderBlock always passes "
            "`query_head_dim_normalize=True`, so queries are scaled by "
            "`head_dim`, not by `query_pre_attn_scalar`."
        )
        self.run_reference_parity_test(
            model_type="gemma3_text",
            reference_class="Gemma3ForCausalLM",
            config=dict(GEMMA3_TEXT_CONFIG, query_pre_attn_scalar=12),
            backbone_cls=Gemma3Backbone,
            task_cls=Gemma3CausalLM,
            known_failures={
                "revealing": (query_scale, 0.66),
                "low_variance": (query_scale, 0.42),
                "epsilon": (EPSILON_IGNORED, 44),
            },
        )
