import unittest
import importlib
import os
import tempfile

from color_normalizer import ColorNormalizer
from evaluate_hybrid_nlp import EVALUATION_SAMPLES
from generate_training_data import generate_bio_training_data, validate_bio_sample
from hybrid_path_nlp import HybridPathNLP
from intent_classifier import IntentClassifier
from slot_tagger import BiLSTMCRFSlotTagger


class FakeIntentClassifier:
    def __init__(self, intent):
        self.intent = intent

    def predict(self, text):
        return self.intent


class FakeSlotTagger:
    def __init__(self, slots):
        self.slots = slots

    def predict(self, text):
        return self.slots


class HybridNLPTest(unittest.TestCase):
    EXPECTED_PARSE_KEYS = {
        "intent",
        "start",
        "waypoints",
        "end",
        "avoid_points",
        "is_complete",
        "missing_slots",
    }

    def test_color_normalizer_maps_aliases(self):
        normalizer = ColorNormalizer()

        self.assertEqual(normalizer.normalize("蓝色点"), "blue")
        self.assertEqual(normalizer.normalize("蓝点"), "blue")
        self.assertEqual(normalizer.normalize("蓝"), "blue")
        self.assertEqual(normalizer.normalize("赤色点"), "red")
        self.assertIsNone(normalizer.normalize("天气"))

    def test_color_normalizer_distinguishes_cyan_and_blue(self):
        normalizer = ColorNormalizer()

        self.assertEqual(normalizer.normalize("青色点"), "cyan")
        self.assertEqual(normalizer.normalize("蓝色点"), "blue")

    def test_intent_classifier_predicts_navigation_and_unknown(self):
        classifier = IntentClassifier()

        self.assertEqual(classifier.predict("帮我规划从紫色点到绿色点的路线"), "navigation")
        self.assertEqual(classifier.predict("今天天气怎么样"), "unknown")

    def test_slot_tagger_extracts_spans_from_bio_tags(self):
        tagger = BiLSTMCRFSlotTagger()

        slots = tagger.decode_slots(
            "从蓝色点经过青色点到绿色点",
            [
                "O",
                "B-START",
                "I-START",
                "I-START",
                "O",
                "O",
                "B-WAYPOINT",
                "I-WAYPOINT",
                "I-WAYPOINT",
                "O",
                "B-END",
                "I-END",
                "I-END",
            ],
        )

        self.assertEqual(
            slots,
            {
                "start_text": "蓝色点",
                "waypoint_texts": ["青色点"],
                "end_text": "绿色点",
                "avoid_texts": [],
            },
        )

    def test_hybrid_parser_uses_model_slots_when_complete(self):
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("navigation"),
            slot_tagger=FakeSlotTagger(
                {
                    "start_text": "蓝色点",
                    "waypoint_texts": ["青色点", "紫点"],
                    "end_text": "绿色点",
                    "avoid_texts": [],
                }
            ),
        )

        self.assertEqual(
            parser.parse("从蓝色点经过青色点和紫点到绿色点"),
            {
                "intent": "navigation",
                "start": "blue",
                "waypoints": ["cyan", "purple"],
                "end": "green",
                "avoid_points": [],
                "is_complete": True,
                "missing_slots": [],
            },
        )

    def test_hybrid_parse_output_fields_are_fixed(self):
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("navigation"),
            slot_tagger=FakeSlotTagger(
                {"start_text": "蓝色点", "waypoint_texts": [], "end_text": "绿色点", "avoid_texts": []}
            ),
        )

        result = parser.parse("从蓝色点到绿色点")

        self.assertEqual(set(result.keys()), self.EXPECTED_PARSE_KEYS)
        self.assertNotIn("used_fallback", result)
        self.assertNotIn("slot_source", result)

    def test_parse_with_debug_reports_model_source(self):
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("navigation"),
            slot_tagger=FakeSlotTagger(
                {"start_text": "蓝色点", "waypoint_texts": [], "end_text": "绿色点", "avoid_texts": []}
            ),
        )

        debug = parser.parse_with_debug("从蓝色点到绿色点")

        self.assertEqual(set(debug["result"].keys()), self.EXPECTED_PARSE_KEYS)
        self.assertFalse(debug["used_fallback"])
        self.assertEqual(debug["slot_source"], "model")
        self.assertIsNone(debug["fallback_reason"])
        self.assertIn("model_loaded", debug)
        self.assertIn("model_path", debug)

    def test_default_debug_reports_model_load_status(self):
        debug = HybridPathNLP().parse_with_debug("从蓝色点到绿色点")

        self.assertIn("model_loaded", debug)
        self.assertIn("model_path", debug)
        if not debug["model_loaded"]:
            self.assertIn("warning", debug)

    def test_hybrid_parser_falls_back_when_slots_are_missing(self):
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("navigation"),
            slot_tagger=FakeSlotTagger(
                {"start_text": None, "waypoint_texts": [], "end_text": None, "avoid_texts": []}
            ),
        )

        self.assertEqual(
            parser.parse("从蓝色点到绿色点"),
            {
                "intent": "navigation",
                "start": "blue",
                "waypoints": [],
                "end": "green",
                "avoid_points": [],
                "is_complete": True,
                "missing_slots": [],
            },
        )

        debug = parser.parse_with_debug("从蓝色点到绿色点")
        self.assertTrue(debug["used_fallback"])
        self.assertEqual(debug["slot_source"], "fallback")
        self.assertEqual(debug["fallback_reason"], "missing_required_slots")

    def test_hybrid_parser_falls_back_when_color_normalization_fails(self):
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("navigation"),
            slot_tagger=FakeSlotTagger(
                {"start_text": "蓝色点", "waypoint_texts": [], "end_text": "火星点", "avoid_texts": []}
            ),
        )

        debug = parser.parse_with_debug("从蓝色点到绿色点")

        self.assertTrue(debug["used_fallback"])
        self.assertEqual(debug["fallback_reason"], "normalization_failed:end")
        self.assertEqual(debug["result"]["end"], "green")

    def test_hybrid_parser_returns_unknown_for_unknown_intent(self):
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("unknown"),
            slot_tagger=FakeSlotTagger({}),
        )

        self.assertEqual(
            parser.parse("今天天气怎么样"),
            {
                "intent": "unknown",
                "start": None,
                "waypoints": [],
                "end": None,
                "avoid_points": [],
                "is_complete": False,
                "missing_slots": [],
            },
        )

    def test_hybrid_parser_keeps_multiple_waypoints_order(self):
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("navigation"),
            slot_tagger=FakeSlotTagger(
                {
                    "start_text": "蓝",
                    "waypoint_texts": ["青色点", "紫色点"],
                    "end_text": "绿",
                    "avoid_texts": [],
                }
            ),
        )

        result = parser.parse("从蓝到绿，途径青色点和紫色点")

        self.assertEqual(result["waypoints"], ["cyan", "purple"])

    def test_hybrid_parser_normalizes_red_aliases(self):
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("navigation"),
            slot_tagger=FakeSlotTagger(
                {"start_text": "红色点", "waypoint_texts": [], "end_text": "赤色点", "avoid_texts": []}
            ),
        )

        result = parser.parse("从红色点到赤色点")

        self.assertEqual(result["start"], "red")
        self.assertEqual(result["end"], "red")

    def test_hybrid_parser_missing_start_slot(self):
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("navigation"),
            slot_tagger=FakeSlotTagger(
                {"start_text": None, "waypoint_texts": [], "end_text": "绿色点", "avoid_texts": []}
            ),
        )

        result = parser.parse("帮我规划到绿色点")

        self.assertIn("start", result["missing_slots"])

    def test_hybrid_parser_missing_end_slot(self):
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("navigation"),
            slot_tagger=FakeSlotTagger(
                {"start_text": "蓝色点", "waypoint_texts": [], "end_text": None, "avoid_texts": []}
            ),
        )

        result = parser.parse("从蓝色点出发")

        self.assertIn("end", result["missing_slots"])

    def test_generate_bio_training_data_contains_valid_bio_sequences(self):
        samples = generate_bio_training_data(limit=5)

        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertEqual(len(sample["text"]), len(sample["tokens"]))
            self.assertEqual(len(sample["tokens"]), len(sample["labels"]))
            self.assertEqual(sample["labels"], sample["tags"])
            self.assertIn("intent", sample)
            self.assertTrue(validate_bio_sample(sample))

    def test_generate_bio_training_data_default_is_large(self):
        samples = generate_bio_training_data()

        self.assertGreaterEqual(len(samples), 1000)

    def test_slot_tagger_save_load_predict_format(self):
        samples = [
            {
                "text": "从蓝色点到绿色点",
                "intent": "navigation",
                "tags": [
                    "O",
                    "B-START",
                    "I-START",
                    "I-START",
                    "O",
                    "B-END",
                    "I-END",
                    "I-END",
                ],
            }
        ]
        tagger = BiLSTMCRFSlotTagger(embedding_dim=8, hidden_dim=16)
        tagger.fit(samples, epochs=1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            model_path = temp_file.name

        try:
            tagger.save(model_path)
            loaded = BiLSTMCRFSlotTagger(model_path=model_path)
            result = loaded.predict("从蓝色点到绿色点")

            self.assertEqual(
                set(result.keys()),
                {"start_text", "waypoint_texts", "end_text", "avoid_texts"},
            )
            self.assertIsInstance(result["waypoint_texts"], list)
            self.assertIsInstance(result["avoid_texts"], list)
            self.assertIn("蓝", loaded.char_to_id)
            self.assertEqual(loaded.id_to_char[loaded.char_to_id["蓝"]], "蓝")
            self.assertEqual(loaded.tag_to_id["B-START"], 1)
            self.assertEqual(loaded.id_to_tag[1], "B-START")
            self.assertEqual(loaded.training_sample_count, 1)
            self.assertEqual(loaded.label_set, list(BiLSTMCRFSlotTagger.TAGS))
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_trained_slot_tagger_can_drive_hybrid_model_path(self):
        samples = [
            {
                "text": "从蓝色点到绿色点",
                "intent": "navigation",
                "tokens": list("从蓝色点到绿色点"),
                "labels": [
                    "O",
                    "B-START",
                    "I-START",
                    "I-START",
                    "O",
                    "B-END",
                    "I-END",
                    "I-END",
                ],
                "tags": [
                    "O",
                    "B-START",
                    "I-START",
                    "I-START",
                    "O",
                    "B-END",
                    "I-END",
                    "I-END",
                ],
            }
        ]
        tagger = BiLSTMCRFSlotTagger(embedding_dim=8, hidden_dim=16)
        tagger.fit(samples, epochs=1)
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("navigation"),
            slot_tagger=tagger,
        )

        debug = parser.parse_with_debug("从蓝色点到绿色点")
        plain = parser.parse("从蓝色点到绿色点")

        self.assertEqual(debug["slot_source"], "model")
        self.assertFalse(debug["used_fallback"])
        self.assertIsNone(debug["fallback_reason"])
        self.assertIn("model_slots", debug)
        self.assertEqual(set(plain.keys()), self.EXPECTED_PARSE_KEYS)

    def test_evaluation_samples_cover_realistic_inputs(self):
        self.assertGreaterEqual(len(EVALUATION_SAMPLES), 80)
        for sample in EVALUATION_SAMPLES:
            self.assertTrue(self.EXPECTED_PARSE_KEYS.issubset(sample.keys()))

    def test_evaluation_samples_include_negative_waypoints(self):
        avoid_samples = [sample for sample in EVALUATION_SAMPLES if "不" in sample["text"] or "避开" in sample["text"] or "绕开" in sample["text"]]

        self.assertGreaterEqual(len(avoid_samples), 6)

    def test_default_hybrid_parser_realistic_cases(self):
        parser = HybridPathNLP()

        cases = [
            ("从蓝色点出发，先经过青色点，再经过紫色点，最后到绿色点", "blue", ["cyan", "purple"], "green", []),
            ("去绿色点，从蓝色点出发", "blue", [], "green", []),
            ("帮我规划到绿色点", None, [], "green", []),
            ("从蓝色点出发", "blue", [], None, []),
            ("从青色点到蓝色点", "cyan", [], "blue", []),
            ("从蓝色点到青色点", "blue", [], "cyan", []),
            ("从蓝色点到绿色点，不经过紫色点", "blue", [], "green", ["purple"]),
            ("我要从蓝色点出发，经过紫，到蓝色，不要经过黄色", "blue", ["purple"], "blue", ["yellow"]),
            ("从蓝到红，避开黄和紫", "blue", [], "red", ["yellow", "purple"]),
        ]

        for text, start, waypoints, end, avoid_points in cases:
            result = parser.parse(text)
            self.assertEqual(set(result.keys()), self.EXPECTED_PARSE_KEYS)
            self.assertEqual(result["start"], start)
            self.assertEqual(result["waypoints"], waypoints)
            self.assertEqual(result["end"], end)
            self.assertEqual(result["avoid_points"], avoid_points)

    def test_default_hybrid_parser_unknown_input(self):
        result = HybridPathNLP().parse("今天天气怎么样")

        self.assertEqual(set(result.keys()), self.EXPECTED_PARSE_KEYS)
        self.assertEqual(result["intent"], "unknown")

    def test_interactive_tools_are_importable(self):
        self.assertIsNotNone(importlib.import_module("interactive_cli"))
        self.assertIsNotNone(importlib.import_module("interactive_gui"))

    def test_negative_training_sample_labels_avoid_color_as_avoid(self):
        samples = generate_bio_training_data()
        sample = next(
            sample
            for sample in samples
            if sample["text"] == "从蓝到绿，不经过紫"
        )
        avoid_index = sample["text"].index("紫")

        self.assertEqual(sample["labels"][avoid_index], "B-AVOID")

    def test_hybrid_parser_outputs_avoid_points_from_model(self):
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("navigation"),
            slot_tagger=FakeSlotTagger(
                {
                    "start_text": "蓝色点",
                    "waypoint_texts": ["紫"],
                    "end_text": "蓝色",
                    "avoid_texts": ["黄色"],
                }
            ),
        )

        debug = parser.parse_with_debug("我要从蓝色点出发，经过紫，到蓝色，不要经过黄色")
        result = debug["result"]

        self.assertEqual(debug["slot_source"], "model")
        self.assertEqual(result["start"], "blue")
        self.assertEqual(result["waypoints"], ["purple"])
        self.assertEqual(result["end"], "blue")
        self.assertEqual(result["avoid_points"], ["yellow"])
        self.assertEqual(debug["model_avoid_points"], ["yellow"])

    def test_hybrid_parser_falls_back_to_fill_missing_avoid(self):
        parser = HybridPathNLP(
            intent_classifier=FakeIntentClassifier("navigation"),
            slot_tagger=FakeSlotTagger(
                {
                    "start_text": "蓝色点",
                    "waypoint_texts": [],
                    "end_text": "绿色点",
                    "avoid_texts": [],
                }
            ),
        )

        debug = parser.parse_with_debug("从蓝色点到绿色点，不经过紫色点")

        self.assertTrue(debug["used_fallback"])
        self.assertEqual(debug["fallback_reason"], "missing_avoid_from_avoid_expression")
        self.assertEqual(debug["result"]["waypoints"], [])
        self.assertEqual(debug["result"]["avoid_points"], ["purple"])


if __name__ == "__main__":
    unittest.main()
