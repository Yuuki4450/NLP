import os
from typing import Dict, List, Optional, Tuple

from color_normalizer import ColorNormalizer
from intent_classifier import IntentClassifier
from path_nlp_parser import PathNLPParser
from slot_tagger import BiLSTMCRFSlotTagger


class HybridPathNLP:
    """Model-first path parser with rule-based fallback."""

    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        slot_tagger: Optional[BiLSTMCRFSlotTagger] = None,
        fallback_parser: Optional[PathNLPParser] = None,
        color_normalizer: Optional[ColorNormalizer] = None,
        model_path: str = "slot_tagger.pkl",
    ):
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.model_loaded = False
        self.model_path = None
        self.model_warning = None
        if slot_tagger is not None:
            self.slot_tagger = slot_tagger
            self.model_loaded = getattr(slot_tagger, "model", None) is not None
            self.model_path = None
        elif model_path and os.path.exists(model_path):
            try:
                self.slot_tagger = BiLSTMCRFSlotTagger(model_path=model_path)
                self.model_loaded = True
                self.model_path = model_path
            except Exception as error:
                self.slot_tagger = BiLSTMCRFSlotTagger()
                self.model_warning = (
                    f"failed to load {model_path}: {type(error).__name__}; "
                    "using untrained slot tagger and fallback may be used frequently"
                )
        else:
            self.slot_tagger = BiLSTMCRFSlotTagger()
            self.model_warning = (
                "slot_tagger.pkl not found; using untrained slot tagger and fallback may be used frequently"
            )
        self.fallback_parser = fallback_parser or PathNLPParser()
        self.color_normalizer = color_normalizer or ColorNormalizer()

    def model_status(self) -> Dict[str, object]:
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "warning": self.model_warning,
        }

    def parse(self, text: str) -> Dict[str, object]:
        return self.parse_with_debug(text)["result"]

    def parse_with_debug(self, text: str) -> Dict[str, object]:
        intent = self.intent_classifier.predict(text)
        if intent == "unknown":
            return {
                "result": self._unknown_result(),
                "used_fallback": False,
                "fallback_reason": None,
                "slot_source": "none",
                "model_slots": None,
                "model_avoid_points": [],
                **self.model_status(),
            }

        if not self._is_navigation_like(intent):
            return self._fallback_result(text, "unsupported_intent", None)

        fallback_preview = self.fallback_parser.parse(text)
        if fallback_preview["intent"] == "unknown":
            return {
                "result": fallback_preview,
                "used_fallback": True,
                "fallback_reason": "fallback_unknown_intent",
                "slot_source": "fallback",
                "model_slots": None,
                "model_avoid_points": [],
                **self.model_status(),
            }

        try:
            model_result, fallback_reason, model_slots = self._parse_with_slot_model(text)
        except Exception as error:
            return self._fallback_result(text, f"slot_tagger_exception:{type(error).__name__}", None)

        if model_result is not None:
            return {
                "result": model_result,
                "used_fallback": False,
                "fallback_reason": None,
                "slot_source": "model",
                "model_slots": model_slots,
                "model_avoid_points": model_result.get("avoid_points", []),
                **self.model_status(),
            }

        return self._fallback_result(text, fallback_reason, model_slots)

    def _parse_with_slot_model(
        self, text: str
    ) -> Tuple[Optional[Dict[str, object]], str, Optional[Dict[str, object]]]:
        slots = self.slot_tagger.predict(text)
        if not isinstance(slots, dict):
            return None, "invalid_slot_output", None

        start_text = slots.get("start_text")
        end_text = slots.get("end_text")
        waypoint_texts = slots.get("waypoint_texts", [])
        avoid_texts = slots.get("avoid_texts", [])

        if waypoint_texts is None:
            waypoint_texts = []
        if avoid_texts is None:
            avoid_texts = []
        if not isinstance(waypoint_texts, list):
            return None, "invalid_slot_output", slots
        if not isinstance(avoid_texts, list):
            return None, "invalid_slot_output", slots
        if not start_text and not end_text and not waypoint_texts and not avoid_texts:
            return None, "missing_required_slots", slots

        start = self.color_normalizer.normalize(start_text) if start_text else None
        end = self.color_normalizer.normalize(end_text) if end_text else None
        waypoints, unmapped_waypoint = self._normalize_waypoints(waypoint_texts, text)
        avoid_points, unmapped_avoid = self._normalize_points(avoid_texts)

        if start_text and start is None:
            return None, "normalization_failed:start", slots
        if end_text and end is None:
            return None, "normalization_failed:end", slots
        if unmapped_waypoint:
            return None, "normalization_failed:waypoint", slots
        if unmapped_avoid:
            return None, "normalization_failed:avoid", slots
        if start is None and end is None and not waypoints and not avoid_points:
            return None, "missing_required_slots", slots
        if self._text_has_avoid_expression(text):
            rule_avoid_points = getattr(
                self.fallback_parser,
                "_extract_avoid_points",
                lambda _: [],
            )(text)
            if not avoid_points:
                return None, "missing_avoid_from_avoid_expression", slots
            if rule_avoid_points and avoid_points != rule_avoid_points:
                return None, "avoid_mismatch_with_rule", slots
        if start is None and self._text_has_start_expression(text):
            return None, "missing_start_from_start_expression", slots
        if end is None and self._text_has_end_expression(text):
            return None, "missing_end_from_end_expression", slots

        missing_slots = []
        if start is None:
            missing_slots.append("start")
        if end is None:
            missing_slots.append("end")

        return {
            "intent": "navigation",
            "start": start,
            "waypoints": waypoints,
            "end": end,
            "avoid_points": avoid_points,
            "is_complete": not missing_slots,
            "missing_slots": missing_slots,
        }, "", slots

    def _normalize_waypoints(self, waypoint_texts: List[str], text: str) -> Tuple[List[str], bool]:
        waypoints, unmapped = self._normalize_points(waypoint_texts)
        if unmapped:
            return waypoints, unmapped
        ignored_avoid_points = self._avoid_colors(text)
        return [waypoint for waypoint in waypoints if waypoint not in ignored_avoid_points], False

    def _normalize_points(self, point_texts: List[str]) -> Tuple[List[str], bool]:
        points = []
        for point_text in point_texts:
            if not point_text or not isinstance(point_text, str):
                return points, True
            point = self.color_normalizer.normalize(point_text)
            if point is None:
                return points, True
            points.append(point)
        return points, False

    def _is_navigation_like(self, intent: str) -> bool:
        is_navigation_like = getattr(self.intent_classifier, "is_navigation_like", None)
        if callable(is_navigation_like):
            return is_navigation_like(intent)
        return intent in {"navigation", "add_waypoint", "query_route"}

    @staticmethod
    def _text_has_start_expression(text: str) -> bool:
        return "从" in text or "起点" in text or "现在在" in text or "我在" in text

    @staticmethod
    def _text_has_end_expression(text: str) -> bool:
        return "到" in text or "去" in text or "终点" in text or "最后" in text

    @staticmethod
    def _text_has_avoid_expression(text: str) -> bool:
        return any(
            keyword in text
            for keyword in (
                "不经过",
                "不要经过",
                "别经过",
                "不路过",
                "不要路过",
                "避开",
                "绕开",
                "不走",
                "不要走",
            )
        )

    def _avoid_colors(self, text: str) -> set:
        colors = set()
        avoid_ranges = getattr(self.fallback_parser, "_avoid_ranges", lambda _: [])(text)
        find_colors = getattr(self.fallback_parser, "_find_colors", None)
        if not callable(find_colors):
            return colors
        for mention in find_colors(text):
            if any(mention.start >= start and mention.end <= end for start, end in avoid_ranges):
                colors.add(mention.color)
        return colors

    def _fallback_result(
        self, text: str, reason: str, model_slots: Optional[Dict[str, object]]
    ) -> Dict[str, object]:
        return {
            "result": self.fallback_parser.parse(text),
            "used_fallback": True,
            "fallback_reason": reason,
            "slot_source": "fallback",
            "model_slots": model_slots,
            "model_avoid_points": self._model_avoid_points(model_slots),
            **self.model_status(),
        }

    @staticmethod
    def _unknown_result() -> Dict[str, object]:
        return {
            "intent": "unknown",
            "start": None,
            "waypoints": [],
            "end": None,
            "avoid_points": [],
            "is_complete": False,
            "missing_slots": [],
        }

    def _model_avoid_points(self, model_slots: Optional[Dict[str, object]]) -> List[str]:
        if not isinstance(model_slots, dict):
            return []
        avoid_texts = model_slots.get("avoid_texts") or []
        if not isinstance(avoid_texts, list):
            return []
        avoid_points, unmapped = self._normalize_points(avoid_texts)
        return [] if unmapped else avoid_points
