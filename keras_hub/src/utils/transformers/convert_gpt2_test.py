import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = GPT2CausalLM.from_preset("hf://openai-community/gpt2")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://openai-community/gpt2",
            load_weights=False,
        )
        self.assertIsInstance(model, GPT2CausalLM)
        model = Backbone.from_preset(
            "hf://openai-community/gpt2",
            load_weights=False,
        )
        self.assertIsInstance(model, GPT2Backbone)

    def test_reference_parity(self):
        self.run_reference_parity_test(
            model_type="gpt2",
            reference_class="GPT2LMHeadModel",
            config={
                "vocab_size": 211,
                "n_positions": 64,
                "n_embd": 48,
                "n_layer": 3,
                "n_head": 4,
                # One id for both, as in GPT-2; the default is outside this
                # vocabulary.
                "bos_token_id": 1,
                "eos_token_id": 1,
            },
            backbone_cls=GPT2Backbone,
            task_cls=GPT2CausalLM,
            known_failures={
                "epsilon": (
                    "convert_gpt2 does not map `layer_norm_epsilon`, and "
                    "GPT2Backbone hard-codes 1e-5.",
                    24,
                ),
            },
        )
