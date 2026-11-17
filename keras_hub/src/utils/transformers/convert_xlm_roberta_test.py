import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.models.xlm_roberta.xlm_roberta_backbone import (
    XLMRobertaBackbone,
)
from keras_hub.src.models.xlm_roberta.xlm_roberta_text_classifier import (
    XLMRobertaTextClassifier,
)
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = XLMRobertaTextClassifier.from_preset(
            "hf://FacebookAI/xlm-roberta-base", num_classes=2
        )
        prompt = "That movie was terrible."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            "hf://FacebookAI/xlm-roberta-base",
            num_classes=2,
            load_weights=False,
        )
        self.assertIsInstance(model, XLMRobertaTextClassifier)
        model = Backbone.from_preset(
            "hf://FacebookAI/xlm-roberta-base",
            load_weights=False,
        )
        self.assertIsInstance(model, XLMRobertaBackbone)

    def test_reference_parity(self):
        epsilon = (
            "convert_xlm_roberta does not map `layer_norm_eps`, and "
            "XLMRobertaBackbone inherits the 1e-5 that RobertaBackbone "
            "hard-codes."
        )
        self.run_reference_parity_test(
            model_type="xlm-roberta",
            reference_class="XLMRobertaModel",
            config={
                "vocab_size": 211,
                "hidden_size": 48,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "intermediate_size": 80,
                "max_position_embeddings": 66,
                "type_vocab_size": 1,
                "pad_token_id": 1,
            },
            backbone_cls=XLMRobertaBackbone,
            ignored_reference_tensors={
                "pooler.dense.*": (
                    "The pooler of XLMRobertaModel. XLMRobertaBackbone has no "
                    "pooled output, and no keras-hub XLM-RoBERTa task uses "
                    "one."
                ),
            },
            known_failures={
                "low_variance": (epsilon, 0.16),
                "epsilon": (epsilon, 6.9),
            },
        )
