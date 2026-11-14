import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mpnet.mpnet_backbone import MPNetBackbone
from keras_hub.src.models.mpnet.mpnet_text_embedder import MPNetTextEmbedder
from keras_hub.src.models.text_embedder import TextEmbedder
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_preset(self):
        model = MPNetTextEmbedder.from_preset(
            "hf://sentence-transformers/all-mpnet-base-v2"
        )
        embeddings = model.encode_text(["That movie was terrible."])
        self.assertEqual(embeddings.shape, (1, 768))

    @pytest.mark.large
    def test_class_detection(self):
        model = TextEmbedder.from_preset(
            "hf://sentence-transformers/all-mpnet-base-v2",
            load_weights=False,
        )
        self.assertIsInstance(model, MPNetTextEmbedder)
        # Pipeline settings come from the sentence-transformer config files.
        self.assertEqual(model.pooling_mode, "mean")
        self.assertTrue(model.normalize)
        self.assertEqual(model.preprocessor.sequence_length, 384)
        self.assertTrue(model.preprocessor.tokenizer.strip_accents)
        model = Backbone.from_preset(
            "hf://sentence-transformers/all-mpnet-base-v2",
            load_weights=False,
        )
        self.assertIsInstance(model, MPNetBackbone)
        self.assertEqual(model.max_sequence_length, 512)

    @pytest.mark.large
    def test_unnormalized_presets(self):
        # The NLI and paraphrase checkpoints have no `Normalize` module and
        # set their own `max_seq_length`.
        for hf_preset, sequence_length in (
            ("hf://sentence-transformers/nli-mpnet-base-v2", 75),
            ("hf://sentence-transformers/paraphrase-mpnet-base-v2", 512),
        ):
            model = TextEmbedder.from_preset(hf_preset, load_weights=False)
            self.assertIsInstance(model, MPNetTextEmbedder)
            self.assertEqual(model.pooling_mode, "mean")
            self.assertFalse(model.normalize)
            self.assertEqual(
                model.preprocessor.sequence_length, sequence_length
            )

    def test_reference_parity(self):
        self.run_reference_parity_test(
            model_type="mpnet",
            reference_class="MPNetModel",
            config={
                "vocab_size": 211,
                "hidden_size": 48,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "intermediate_size": 80,
                "max_position_embeddings": 64,
                "pad_token_id": 1,
            },
            backbone_cls=MPNetBackbone,
            ignored_reference_tensors={
                "pooler.dense.*": (
                    "The pooler of MPNetModel. MPNetBackbone has no pooled "
                    "output; MPNetTextEmbedder pools the sequence output, as "
                    "sentence-transformers does."
                ),
            },
        )
