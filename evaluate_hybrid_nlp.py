import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional

from generate_training_data import generate_bio_training_data
from hybrid_path_nlp import HybridPathNLP
from slot_tagger import BiLSTMCRFSlotTagger


def nav(
    text: str,
    start: Optional[str],
    waypoints: List[str],
    end: Optional[str],
    avoid_points: Optional[List[str]] = None,
) -> Dict[str, object]:
    missing_slots = []
    if start is None:
        missing_slots.append("start")
    if end is None:
        missing_slots.append("end")
    return {
        "text": text,
        "intent": "navigation",
        "start": start,
        "waypoints": waypoints,
        "end": end,
        "avoid_points": list(avoid_points or []),
        "is_complete": not missing_slots,
        "missing_slots": missing_slots,
    }


def unknown(text: str) -> Dict[str, object]:
    return {
        "text": text,
        "intent": "unknown",
        "start": None,
        "waypoints": [],
        "end": None,
        "avoid_points": [],
        "is_complete": False,
        "missing_slots": [],
    }


EVALUATION_SAMPLES: List[Dict[str, object]] = [
    # Standard path expressions.
    nav("从蓝色点到绿色点", "blue", [], "green"),
    nav("从红色点到紫色点", "red", [], "purple"),
    nav("从橙色点去青色点", "orange", [], "cyan"),
    nav("我要从黄色点到蓝色点", "yellow", [], "blue"),
    nav("帮我规划从紫色点到红色点的路线", "purple", [], "red"),
    nav("从绿色点到橙色点", "green", [], "orange"),
    nav("从青色点去黄色点", "cyan", [], "yellow"),
    nav("我要从蓝点到赤点", "blue", [], "red"),
    nav("帮我规划从橙点去紫点的路线", "orange", [], "purple"),
    nav("从黄点到绿点", "yellow", [], "green"),
    # Short color aliases.
    nav("蓝到绿", "blue", [], "green"),
    nav("红去紫", "red", [], "purple"),
    nav("青到橙", "cyan", [], "orange"),
    nav("黄点到蓝点", "yellow", [], "blue"),
    nav("赤到青", "red", [], "cyan"),
    nav("紫去红", "purple", [], "red"),
    nav("绿到黄", "green", [], "yellow"),
    nav("橙去蓝", "orange", [], "blue"),
    nav("蓝点去赤点", "blue", [], "red"),
    nav("青点到紫点", "cyan", [], "purple"),
    # Inverted start/end expressions.
    nav("去绿色点，从蓝色点出发", "blue", [], "green"),
    nav("到紫色点，起点是红色点", "red", [], "purple"),
    nav("终点是青色点，起点是橙色点", "orange", [], "cyan"),
    nav("目标是绿色点，我从蓝色点出发", "blue", [], "green"),
    nav("到蓝点，从黄点出发", "yellow", [], "blue"),
    nav("去赤色点，起点是紫色点", "purple", [], "red"),
    nav("终点是黄色点，起点是绿色点", "green", [], "yellow"),
    nav("目标是紫点，我从青点出发", "cyan", [], "purple"),
    # One waypoint.
    nav("从蓝色点到绿色点，经过青色点", "blue", ["cyan"], "green"),
    nav("从红点去紫点，途径黄点", "red", ["yellow"], "purple"),
    nav("从橙色点出发，先去青色点，最后到蓝色点", "orange", ["cyan"], "blue"),
    nav("蓝到绿，中间经过紫", "blue", ["purple"], "green"),
    nav("从赤点到青点，路过橙点", "red", ["orange"], "cyan"),
    nav("从黄点经过蓝点到绿点", "yellow", ["blue"], "green"),
    nav("从紫色点途径红色点到青色点", "purple", ["red"], "cyan"),
    nav("绿点到橙点，经过蓝点", "green", ["blue"], "orange"),
    nav("从青点出发，经过黄点，最后到红点", "cyan", ["yellow"], "red"),
    nav("从蓝色点到黄色点，中间经过绿色点", "blue", ["green"], "yellow"),
    # Multiple waypoints.
    nav("从蓝到红，途径黄和紫", "blue", ["yellow", "purple"], "red"),
    nav(
        "从蓝色点出发，先经过青色点，再经过紫色点，最后到绿色点",
        "blue",
        ["cyan", "purple"],
        "green",
    ),
    nav(
        "红色点到蓝色点，中间依次经过橙色点、黄色点、绿色点",
        "red",
        ["orange", "yellow", "green"],
        "blue",
    ),
    nav("从紫点到青点，先去红点，再去黄点", "purple", ["red", "yellow"], "cyan"),
    nav(
        "从橙色点出发，经过蓝色点，再经过绿色点，最后到赤色点",
        "orange",
        ["blue", "green"],
        "red",
    ),
    nav("从红点先去橙点再去黄点最后到蓝点", "red", ["orange", "yellow"], "blue"),
    nav("从绿点经过青点再经过紫点到黄点", "green", ["cyan", "purple"], "yellow"),
    nav("蓝点到红点，先去青点，再去橙点", "blue", ["cyan", "orange"], "red"),
    nav("从紫色点到绿色点，途经红色点和蓝色点", "purple", ["red", "blue"], "green"),
    nav("从黄点到紫点，中间依次经过蓝点、青点", "yellow", ["blue", "cyan"], "purple"),
    # No punctuation.
    nav("从蓝色点经过青色点到绿色点", "blue", ["cyan"], "green"),
    nav("从红点先去橙点再去黄点最后到蓝点", "red", ["orange", "yellow"], "blue"),
    nav("我要从紫点经过青点到红点", "purple", ["cyan"], "red"),
    nav("蓝点出发经过黄点最后到绿点", "blue", ["yellow"], "green"),
    nav("从橙点途径青点到紫点", "orange", ["cyan"], "purple"),
    nav("红点到蓝点经过绿点", "red", ["green"], "blue"),
    nav("紫点出发先去红点再到青点", "purple", ["red"], "cyan"),
    nav("从黄点路过橙点到蓝点", "yellow", ["orange"], "blue"),
    # Colloquial expressions.
    nav("我想从蓝色点走到绿色点", "blue", [], "green"),
    nav("能不能帮我从红点导航到紫点", "red", [], "purple"),
    nav("帮我找一条蓝点到绿点的路", "blue", [], "green"),
    nav("我在青色点，想去红色点", "cyan", [], "red"),
    nav("从橙色点开始，最后去蓝色点", "orange", [], "blue"),
    nav("我在紫点，想去黄点", "purple", [], "yellow"),
    nav("帮我找一条红点到青点的路", "red", [], "cyan"),
    nav("能不能帮我从蓝点导航到橙点", "blue", [], "orange"),
    nav("我想从绿点走到赤点", "green", [], "red"),
    nav("从黄色点开始，最后去紫色点", "yellow", [], "purple"),
    # Missing start.
    nav("去绿色点", None, [], "green"),
    nav("到红点", None, [], "red"),
    nav("帮我规划到紫色点", None, [], "purple"),
    nav("终点是青色点", None, [], "cyan"),
    nav("我想去蓝点", None, [], "blue"),
    nav("最后到橙点", None, [], "orange"),
    nav("去赤色点", None, [], "red"),
    nav("到黄点", None, [], "yellow"),
    # Missing end.
    nav("从蓝色点出发", "blue", [], None),
    nav("起点是红色点", "red", [], None),
    nav("我现在在青点", "cyan", [], None),
    nav("从紫点开始", "purple", [], None),
    nav("我在橙色点", "orange", [], None),
    nav("从黄点出发", "yellow", [], None),
    nav("起点是蓝点", "blue", [], None),
    nav("我现在在赤点", "red", [], None),
    # Non-navigation.
    unknown("今天天气怎么样"),
    unknown("我想听音乐"),
    unknown("你好"),
    unknown("帮我查新闻"),
    unknown("这个地图好看吗"),
    unknown("今天吃什么"),
    unknown("打开音乐"),
    unknown("讲个笑话"),
    # Confusing color cases and avoid constraints.
    nav("从青色点到蓝色点", "cyan", [], "blue"),
    nav("从蓝色点到青色点", "blue", [], "cyan"),
    nav("从赤色点到红色点", "red", [], "red"),
    nav("从红色点到赤色点", "red", [], "red"),
    nav("从绿色点到青色点，经过蓝色点", "green", ["blue"], "cyan"),
    nav("从蓝色点到绿色点，不经过紫色点", "blue", [], "green", ["purple"]),
    nav("我要从蓝色点出发，经过紫，到蓝色，不要经过黄色", "blue", ["purple"], "blue", ["yellow"]),
    nav("从红点到蓝点，途径橙点，避开黄色点", "red", ["orange"], "blue", ["yellow"]),
    nav("从青点到紫点，先经过绿点，不要路过橙点", "cyan", ["green"], "purple", ["orange"]),
    nav("从蓝到红，绕开黄，经过紫", "blue", ["purple"], "red", ["yellow"]),
    nav("从橙到青，不走红，经过绿", "orange", ["green"], "cyan", ["red"]),
    nav("从蓝到红，避开黄和紫", "blue", [], "red", ["yellow", "purple"]),
    nav("去绿色点，避开红点", None, [], "green", ["red"]),
    nav("从蓝色点出发，不经过黄色点", "blue", [], None, ["yellow"]),
    nav("从青点到蓝点，经过红点", "cyan", ["red"], "blue"),
    nav("从蓝点到青点，经过赤点", "blue", ["red"], "cyan"),
    # Abnormal input.
    unknown(""),
    unknown("   "),
    unknown("蓝色点"),
    unknown("红"),
    unknown("从到"),
    nav("从蓝到", "blue", [], None),
]


EXPECTED_PARSE_KEYS = {
    "intent",
    "start",
    "waypoints",
    "end",
    "avoid_points",
    "is_complete",
    "missing_slots",
}


def evaluate(parser: HybridPathNLP, samples: List[Dict[str, object]]) -> Dict[str, object]:
    counters = {
        "intent": 0,
        "start": 0,
        "end": 0,
        "waypoints": 0,
        "avoid_points": 0,
        "is_complete": 0,
        "missing_slots": 0,
        "semantic_frame": 0,
        "fallback": 0,
    }
    slot_source_counts = Counter()
    fallback_reason_counts = Counter()
    fallback_examples = []
    errors = []

    for sample in samples:
        debug = _parse_with_debug(parser, sample["text"])
        predicted = debug["result"]
        slot_source_counts[debug.get("slot_source", "unknown")] += 1
        if debug["used_fallback"]:
            counters["fallback"] += 1
            reason = debug.get("fallback_reason") or "unknown"
            fallback_reason_counts[reason] += 1
            fallback_examples.append(
                {
                    "text": sample["text"],
                    "model_slots": debug.get("model_slots"),
                    "fallback_reason": reason,
                    "final_result": predicted,
                }
            )

        error_fields = []
        for field, counter_name in [
            ("intent", "intent"),
            ("start", "start"),
            ("end", "end"),
            ("waypoints", "waypoints"),
            ("avoid_points", "avoid_points"),
            ("is_complete", "is_complete"),
            ("missing_slots", "missing_slots"),
        ]:
            if predicted[field] == sample[field]:
                counters[counter_name] += 1
            else:
                error_fields.append(field)

        if (
            predicted["start"] == sample["start"]
            and predicted["waypoints"] == sample["waypoints"]
            and predicted["end"] == sample["end"]
            and predicted["avoid_points"] == sample["avoid_points"]
        ):
            counters["semantic_frame"] += 1

        if error_fields:
            errors.append(
                {
                    "text": sample["text"],
                    "expected": _expected_result(sample),
                    "predicted": predicted,
                    "error_fields": error_fields,
                    "fallback_reason": debug.get("fallback_reason"),
                }
            )

    total = len(samples)
    return {
        "total_samples": total,
        "intent_accuracy": counters["intent"] / total,
        "start_accuracy": counters["start"] / total,
        "end_accuracy": counters["end"] / total,
        "waypoint_exact_match_accuracy": counters["waypoints"] / total,
        "avoid_exact_match_accuracy": counters["avoid_points"] / total,
        "is_complete_accuracy": counters["is_complete"] / total,
        "missing_slots_accuracy": counters["missing_slots"] / total,
        "semantic_frame_accuracy": counters["semantic_frame"] / total,
        "fallback_count": counters["fallback"],
        "fallback_rate": counters["fallback"] / total,
        "slot_source_counts": dict(slot_source_counts),
        "fallback_reason_counts": dict(fallback_reason_counts),
        "fallback_examples": fallback_examples,
        "errors": errors,
    }


def print_report(metrics: Dict[str, object], verbose: bool = False) -> None:
    total = metrics["total_samples"]
    print(f"Total samples: {total}")
    print(f"Intent accuracy: {metrics['intent_accuracy']:.4f}")
    print(f"Start accuracy: {metrics['start_accuracy']:.4f}")
    print(f"End accuracy: {metrics['end_accuracy']:.4f}")
    print(f"Waypoint exact match accuracy: {metrics['waypoint_exact_match_accuracy']:.4f}")
    print(f"Avoid exact match accuracy: {metrics['avoid_exact_match_accuracy']:.4f}")
    print(f"Is complete accuracy: {metrics['is_complete_accuracy']:.4f}")
    print(f"Missing slots accuracy: {metrics['missing_slots_accuracy']:.4f}")
    print(f"Semantic frame accuracy: {metrics['semantic_frame_accuracy']:.4f}")
    print(f"Fallback used: {metrics['fallback_count']} / {total}")
    print(f"Fallback rate: {metrics['fallback_rate']:.4f}")
    print(f"Slot source model: {metrics['slot_source_counts'].get('model', 0)}")
    print(f"Slot source fallback: {metrics['slot_source_counts'].get('fallback', 0)}")
    print("Fallback reason counts:")
    if metrics["fallback_reason_counts"]:
        for reason, count in sorted(metrics["fallback_reason_counts"].items()):
            print(f"- {reason}: {count}")
    else:
        print("- none: 0")

    if verbose and metrics["fallback_examples"]:
        print("\nFallback examples:")
        for example in metrics["fallback_examples"]:
            print(f"\ntext: {example['text']}")
            print("model slots:", json.dumps(example["model_slots"], ensure_ascii=False))
            print("fallback reason:", example["fallback_reason"])
            print("final result:", json.dumps(example["final_result"], ensure_ascii=False))

    errors = metrics["errors"]
    if not errors:
        print("No prediction errors.")
        return

    print("\nErrors:")
    for index, error in enumerate(errors, start=1):
        print(f"\n[{index}] text: {error['text']}")
        print("expected:", json.dumps(error["expected"], ensure_ascii=False))
        print("predicted:", json.dumps(error["predicted"], ensure_ascii=False))
        print("error fields:", ", ".join(error["error_fields"]))
        if error["fallback_reason"]:
            print("fallback reason:", error["fallback_reason"])


def _parse_with_debug(parser: HybridPathNLP, text: str) -> Dict[str, object]:
    parse_with_debug = getattr(parser, "parse_with_debug", None)
    if callable(parse_with_debug):
        return parse_with_debug(text)
    return {
        "result": parser.parse(text),
        "used_fallback": False,
        "slot_source": "unknown",
        "fallback_reason": None,
        "model_slots": None,
    }


def _expected_result(sample: Dict[str, object]) -> Dict[str, object]:
    return {
        "intent": sample["intent"],
        "start": sample["start"],
        "waypoints": sample["waypoints"],
        "end": sample["end"],
        "avoid_points": sample["avoid_points"],
        "is_complete": sample["is_complete"],
        "missing_slots": sample["missing_slots"],
    }


def build_parser(args) -> HybridPathNLP:
    if args.model_path and os.path.exists(args.model_path):
        return HybridPathNLP(model_path=args.model_path)

    if args.no_eval_slot_training:
        return HybridPathNLP()

    samples = generate_bio_training_data(limit=args.eval_train_limit)
    tagger = BiLSTMCRFSlotTagger(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    tagger.fit(samples, epochs=args.eval_train_epochs, learning_rate=args.lr)
    return HybridPathNLP(slot_tagger=tagger)


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Evaluate HybridPathNLP.")
    arg_parser.add_argument("--model-path", default="slot_tagger.pkl")
    arg_parser.add_argument("--no-eval-slot-training", action="store_true")
    arg_parser.add_argument("--eval-train-limit", type=int, default=None)
    arg_parser.add_argument("--eval-train-epochs", type=int, default=0)
    arg_parser.add_argument("--embedding-dim", type=int, default=32)
    arg_parser.add_argument("--hidden-dim", type=int, default=64)
    arg_parser.add_argument("--dropout", type=float, default=0.0)
    arg_parser.add_argument("--lr", type=float, default=0.005)
    arg_parser.add_argument("--verbose", action="store_true")
    args = arg_parser.parse_args()

    parser = build_parser(args)
    print_report(evaluate(parser, EVALUATION_SAMPLES), verbose=args.verbose)


if __name__ == "__main__":
    main()
