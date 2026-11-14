import pytest

from keras_hub.src.models.albert.albert_backbone import AlbertBackbone
from keras_hub.src.models.albert.albert_text_classifier import (
    AlbertTextClassifier,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = AlbertTextClassifier.from_preset(
            "hf://albert/albert-base-v2", num_classes=2
        )
        prompt = "That movies was terrible."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            "hf://albert/albert-base-v2",
            num_classes=2,
            load_weights=False,
        )
        self.assertIsInstance(model, AlbertTextClassifier)
        model = Backbone.from_preset(
            "hf://albert/albert-base-v2",
            load_weights=False,
        )
        self.assertIsInstance(model, AlbertBackbone)

    def test_reference_parity(self):
        self.run_reference_parity_test(
            model_type="albert",
            reference_class="AlbertForPreTraining",
            config={
                "vocab_size": 211,
                "embedding_size": 24,
                "hidden_size": 48,
                "num_hidden_layers": 4,
                "num_hidden_groups": 2,
                "inner_group_num": 2,
                "num_attention_heads": 4,
                "intermediate_size": 80,
                "max_position_embeddings": 64,
                "type_vocab_size": 2,
                "pad_token_id": 0,
            },
            backbone_cls=AlbertBackbone,
            ignored_reference_tensors={
                "predictions.*": (
                    "The masked-LM head of AlbertForPreTraining; "
                    "AlbertBackbone has no head."
                ),
                "sop_classifier.*": (
                    "The sentence-order prediction head of "
                    "AlbertForPreTraining; keras-hub has no such head."
                ),
            },
            known_failures={
                "epsilon": (
                    "convert_albert does not map `layer_norm_eps`, and "
                    "AlbertBackbone hard-codes 1e-12.",
                    6.5,
                ),
            },
        )
