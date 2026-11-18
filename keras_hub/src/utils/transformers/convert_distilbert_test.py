import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.distil_bert.distil_bert_backbone import (
    DistilBertBackbone,
)
from keras_hub.src.models.distil_bert.distil_bert_text_classifier import (
    DistilBertTextClassifier,
)
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = DistilBertTextClassifier.from_preset(
            "hf://distilbert/distilbert-base-uncased", num_classes=2
        )
        prompt = "That movies was terrible."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            "hf://distilbert/distilbert-base-uncased",
            num_classes=2,
            load_weights=False,
        )
        self.assertIsInstance(model, DistilBertTextClassifier)
        model = Backbone.from_preset(
            "hf://distilbert/distilbert-base-uncased",
            load_weights=False,
        )
        self.assertIsInstance(model, DistilBertBackbone)

    def test_reference_parity(self):
        self.run_reference_parity_test(
            model_type="distilbert",
            # convert_distilbert reads keys under `distilbert.`, which the task
            # classes save and DistilBertModel does not.
            reference_class="DistilBertForMaskedLM",
            config={
                "vocab_size": 211,
                "dim": 48,
                "n_layers": 3,
                "n_heads": 4,
                "hidden_dim": 80,
                "max_position_embeddings": 64,
                "pad_token_id": 0,
            },
            cls=DistilBertBackbone,
            ignored_reference_tensors={
                "vocab_*": (
                    "The masked-LM head of DistilBertForMaskedLM; "
                    "DistilBertBackbone has no head."
                ),
            },
        )
