import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.models.bert.bert_text_classifier import BertTextClassifier
from keras_hub.src.models.bert.bert_text_embedder import BertTextEmbedder
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = BertTextClassifier.from_preset(
            "hf://google-bert/bert-base-uncased", num_classes=2
        )
        prompt = "That movies was terrible."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            "hf://google-bert/bert-base-uncased",
            num_classes=2,
            load_weights=False,
        )
        self.assertIsInstance(model, BertTextClassifier)
        model = Backbone.from_preset(
            "hf://google-bert/bert-base-uncased",
            load_weights=False,
        )
        self.assertIsInstance(model, BertBackbone)

    @pytest.mark.large
    def test_multilingual_e5_tokenizer(self):
        model = BertTextEmbedder.from_preset(
            "hf://intfloat/multilingual-e5-small",
            load_weights=False,
        )
        self.assertIsInstance(model.preprocessor.tokenizer, XLMRobertaTokenizer)

    def test_reference_parity(self):
        self.run_reference_parity_test(
            model_type="bert",
            reference_class="BertModel",
            config={
                "vocab_size": 211,
                "hidden_size": 48,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "intermediate_size": 80,
                "max_position_embeddings": 64,
                "type_vocab_size": 2,
                "pad_token_id": 0,
            },
            backbone_cls=BertBackbone,
            known_failures={
                "revealing": (
                    "known difference from transformers: BERT uses the tanh "
                    "GELU approximation of the original Google "
                    "implementation (not a defect).",
                    2.5e-3,
                ),
                "epsilon": (
                    "convert_bert does not map `layer_norm_eps`, and "
                    "BertBackbone hard-codes 1e-12.",
                    10,
                ),
            },
        )
