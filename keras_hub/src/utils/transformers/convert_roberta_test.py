import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.roberta.roberta_backbone import RobertaBackbone
from keras_hub.src.models.roberta.roberta_text_classifier import (
    RobertaTextClassifier,
)
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.tests.test_case import TestCase


class TestRobertaConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = RobertaTextClassifier.from_preset(
            "hf://FacebookAI/roberta-base", num_classes=2
        )
        prompt = "That movies was terrible."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            "hf://FacebookAI/roberta-base",
            num_classes=2,
            load_weights=False,
        )
        self.assertIsInstance(model, RobertaTextClassifier)
        model = Backbone.from_preset(
            "hf://FacebookAI/roberta-base",
            load_weights=False,
        )
        self.assertIsInstance(model, RobertaBackbone)

    def test_reference_parity(self):
        token_type = (
            "RobertaBackbone leaves out the token type embedding that "
            "transformers adds at every position, which hides any other defect."
        )
        epsilon = (
            "The missing token type embedding, and convert_roberta does not "
            "map `layer_norm_eps`, which RobertaBackbone hard-codes as 1e-5."
        )
        self.run_reference_parity_test(
            model_type="roberta",
            reference_class="RobertaForMaskedLM",
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
            backbone_cls=RobertaBackbone,
            ignored_reference_tensors={
                "lm_head.*": (
                    "The masked-LM head of RobertaForMaskedLM; "
                    "RobertaBackbone has no head."
                ),
                "roberta.embeddings.token_type_embeddings.weight": (
                    "convert_roberta never reads it, and RobertaBackbone has "
                    "no token type embedding."
                ),
            },
            # While the token type embedding is left out, every check
            # measures 3 to 5, so none can detect a wiring bug: query and
            # key biases swapped in every layer still give XFAIL. Only the
            # load and its audit are tested.
            known_failures={
                "revealing": (token_type, 8.0),
                "low_variance": (epsilon, 8.0),
                "epsilon": (epsilon, 10),
            },
        )
