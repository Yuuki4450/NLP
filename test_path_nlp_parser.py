import unittest

from path_nlp_parser import PathNLPParser


class PathNLPParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = PathNLPParser()

    def assertParse(self, text, expected):
        expected.setdefault("avoid_points", [])
        self.assertEqual(self.parser.parse(text), expected)

    def test_parse_direct_from_to(self):
        self.assertParse(
            "从蓝色点到绿色点",
            {
                "intent": "navigation",
                "start": "blue",
                "waypoints": [],
                "end": "green",
                "is_complete": True,
                "missing_slots": [],
            },
        )

    def test_parse_want_from_go_to(self):
        self.assertParse(
            "我要从蓝点去绿点",
            {
                "intent": "navigation",
                "start": "blue",
                "waypoints": [],
                "end": "green",
                "is_complete": True,
                "missing_slots": [],
            },
        )

    def test_parse_start_waypoint_end(self):
        self.assertParse(
            "从蓝色出发，经过青色点，到绿色点",
            {
                "intent": "navigation",
                "start": "blue",
                "waypoints": ["cyan"],
                "end": "green",
                "is_complete": True,
                "missing_slots": [],
            },
        )

    def test_parse_waypoints_after_end_keep_user_order(self):
        self.assertParse(
            "从蓝到红，途径黄和紫",
            {
                "intent": "navigation",
                "start": "blue",
                "waypoints": ["yellow", "purple"],
                "end": "red",
                "is_complete": True,
                "missing_slots": [],
            },
        )

    def test_parse_ordered_waypoints_before_end(self):
        self.assertParse(
            "蓝色点到红色点，先经过橙色点，再经过黄色点",
            {
                "intent": "navigation",
                "start": "blue",
                "waypoints": ["orange", "yellow"],
                "end": "red",
                "is_complete": True,
                "missing_slots": [],
            },
        )

    def test_parse_route_planning_sentence(self):
        self.assertParse(
            "帮我规划从紫色点到绿色点的路线",
            {
                "intent": "navigation",
                "start": "purple",
                "waypoints": [],
                "end": "green",
                "is_complete": True,
                "missing_slots": [],
            },
        )

    def test_parse_end_before_start(self):
        self.assertParse(
            "去绿色点，从蓝色点出发",
            {
                "intent": "navigation",
                "start": "blue",
                "waypoints": [],
                "end": "green",
                "is_complete": True,
                "missing_slots": [],
            },
        )

    def test_parse_missing_end(self):
        self.assertParse(
            "从蓝色点出发",
            {
                "intent": "navigation",
                "start": "blue",
                "waypoints": [],
                "end": None,
                "is_complete": False,
                "missing_slots": ["end"],
            },
        )

    def test_parse_missing_start(self):
        self.assertParse(
            "到绿色点",
            {
                "intent": "navigation",
                "start": None,
                "waypoints": [],
                "end": "green",
                "is_complete": False,
                "missing_slots": ["start"],
            },
        )

    def test_parse_unknown_intent(self):
        self.assertParse(
            "今天天气怎么样",
            {
                "intent": "unknown",
                "start": None,
                "waypoints": [],
                "end": None,
                "is_complete": False,
                "missing_slots": [],
            },
        )

    def test_parse_red_aliases(self):
        self.assertParse(
            "从赤点到赤色点",
            {
                "intent": "navigation",
                "start": "red",
                "waypoints": [],
                "end": "red",
                "is_complete": True,
                "missing_slots": [],
            },
        )

    def test_parse_multiple_destination_phrases_as_waypoints_without_start(self):
        self.assertParse(
            "先去青色点，再到紫色点，最后到红色点",
            {
                "intent": "navigation",
                "start": None,
                "waypoints": ["cyan", "purple"],
                "end": "red",
                "is_complete": False,
                "missing_slots": ["start"],
            },
        )

    def test_parse_waypoints_before_final_end_does_not_infer_start(self):
        self.assertParse(
            "先经过橙色点，再经过黄色点，最后到紫色点",
            {
                "intent": "navigation",
                "start": None,
                "waypoints": ["orange", "yellow"],
                "end": "purple",
                "is_complete": False,
                "missing_slots": ["start"],
            },
        )

    def test_parse_target_before_later_go_waypoints(self):
        self.assertParse(
            "从紫点到青点，先去红点，再去黄点",
            {
                "intent": "navigation",
                "start": "purple",
                "waypoints": ["red", "yellow"],
                "end": "cyan",
                "is_complete": True,
                "missing_slots": [],
            },
        )

    def test_parse_ignores_negative_waypoint_phrases(self):
        cases = [
            (
                "从蓝色点到绿色点，不经过紫色点",
                "blue",
                [],
                "green",
                ["purple"],
            ),
            (
                "我要从蓝色点出发，经过紫，到蓝色，不要经过黄色",
                "blue",
                ["purple"],
                "blue",
                ["yellow"],
            ),
            (
                "从红点到蓝点，途径橙点，避开黄色点",
                "red",
                ["orange"],
                "blue",
                ["yellow"],
            ),
            (
                "从青点到紫点，先经过绿点，不要路过橙点",
                "cyan",
                ["green"],
                "purple",
                ["orange"],
            ),
            (
                "从蓝到红，绕开黄，经过紫",
                "blue",
                ["purple"],
                "red",
                ["yellow"],
            ),
            (
                "从蓝到红，避开黄和紫",
                "blue",
                [],
                "red",
                ["yellow", "purple"],
            ),
        ]

        for text, start, waypoints, end, avoid_points in cases:
            result = self.parser.parse(text)
            self.assertEqual(result["start"], start)
            self.assertEqual(result["waypoints"], waypoints)
            self.assertEqual(result["end"], end)
            self.assertEqual(result["avoid_points"], avoid_points)


if __name__ == "__main__":
    unittest.main()
