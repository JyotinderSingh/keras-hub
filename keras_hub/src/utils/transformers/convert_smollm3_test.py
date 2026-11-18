import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.smollm3.smollm3_backbone import SmolLM3Backbone
from keras_hub.src.models.smollm3.smollm3_causal_lm import SmolLM3CausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_smollm3


class TestSmollm3Converter(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = SmolLM3CausalLM.from_preset("hf://yujiepan/smollm3-tiny-random")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.extra_large
    def test_convert_preset(self):
        model = SmolLM3CausalLM.from_preset("hf://HuggingFaceTB/SmolLM3-3B")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.extra_large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://HuggingFaceTB/SmolLM3-3B",
            load_weights=False,
        )
        self.assertIsInstance(model, SmolLM3CausalLM)
        model = Backbone.from_preset(
            "hf://HuggingFaceTB/SmolLM3-3B",
            load_weights=False,
        )
        self.assertIsInstance(model, SmolLM3Backbone)

    def test_convert_backbone_config_rope_theta(self):
        # transformers < 5 format
        transformers_config = {
            "vocab_size": 100,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 48,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "no_rope_layers": [1, 1],
            "layer_types": ["full_attention", "full_attention"],
            "mlp_bias": False,
        }
        keras_config = convert_smollm3.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_theta"], 10000.0)

        # transformers >= 5 format
        transformers_config["rope_parameters"] = {"rope_theta": 20000.0}
        keras_config = convert_smollm3.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_theta"], 20000.0)

    def test_reference_parity(self):
        self.run_reference_parity_test(
            model_type="smollm3",
            reference_class="SmolLM3ForCausalLM",
            config={
                "vocab_size": 211,
                "hidden_size": 72,
                "intermediate_size": 100,
                # SmolLM3 skips RoPE in every fourth layer by default, so
                # four layers cover one layer without RoPE.
                "num_hidden_layers": 4,
                "num_attention_heads": 6,
                "num_key_value_heads": 2,
                "max_position_embeddings": 64,
                # The default special token ids are outside this vocabulary.
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
            cls=SmolLM3CausalLM,
            derived_weights={
                "rotary_emb/inv_freq": (
                    "RoPE inverse frequencies, computed from `rope_theta` "
                    "and the head size when the layer is created; "
                    "transformers keeps them in a non-persistent buffer, so "
                    "checkpoints do not store them."
                ),
            },
        )
