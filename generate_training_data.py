import argparse
import json
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BIO_TAGS = (
    "O",
    "B-START",
    "I-START",
    "B-END",
    "I-END",
    "B-WAYPOINT",
    "I-WAYPOINT",
    "B-AVOID",
    "I-AVOID",
)


COLOR_ALIASES = {
    "red": ["红", "红点", "红色", "红色点", "赤", "赤点", "赤色", "赤色点"],
    "orange": ["橙", "橙点", "橙色", "橙色点"],
    "yellow": ["黄", "黄点", "黄色", "黄色点"],
    "green": ["绿", "绿点", "绿色", "绿色点"],
    "cyan": ["青", "青点", "青色", "青色点"],
    "blue": ["蓝", "蓝点", "蓝色", "蓝色点"],
    "purple": ["紫", "紫点", "紫色", "紫色点"],
}


NO_WAYPOINT_TEMPLATES = (
    "从{START}到{END}",
    "从{START}去{END}",
    "我要从{START}到{END}",
    "我要从{START}去{END}",
    "帮我规划从{START}到{END}的路线",
    "{START}到{END}",
    "{START}去{END}",
    "{START}出发到{END}",
    "从{START}出发，最后到{END}",
    "去{END}，从{START}出发",
    "到{END}，从{START}出发",
    "终点是{END}，起点是{START}",
    "到{END}，起点是{START}",
    "目标是{END}，我从{START}出发",
    "我想从{START}走到{END}",
    "能不能帮我从{START}导航到{END}",
    "帮我找一条{START}到{END}的路",
    "我在{START}，想去{END}",
    "从{START}开始，最后去{END}",
)


ONE_WAYPOINT_TEMPLATES = (
    "从{START}经过{WP}到{END}",
    "从{START}途径{WP}到{END}",
    "从{START}路过{WP}到{END}",
    "从{START}出发，经过{WP}，最后到{END}",
    "从{START}出发，经过{WP}，到{END}",
    "从{START}出发，途经{WP}，去{END}",
    "从{START}到{END}，经过{WP}",
    "从{START}到{END}，中间经过{WP}",
    "从{START}到{END}，途径{WP}",
    "{START}到{END}经过{WP}",
    "{START}出发，先去{WP}，最后到{END}",
    "先去{WP}，再到{END}，从{START}出发",
    "{START}到{END}，路过{WP}",
    "{START}到{END}，中间经过{WP}",
    "{START}出发经过{WP}最后到{END}",
)


TWO_WAYPOINT_TEMPLATES = (
    "从{START}经过{WP1}再经过{WP2}到{END}",
    "从{START}出发，先经过{WP1}，再经过{WP2}，最后到{END}",
    "从{START}到{END}，途径{WP1}和{WP2}",
    "从{START}到{END}，途经{WP1}和{WP2}",
    "从{START}到{END}，经过{WP1}和{WP2}",
    "从{START}到{END}，中间依次经过{WP1}、{WP2}",
    "{START}到{END}，先经过{WP1}，再经过{WP2}",
    "{START}到{END}，先去{WP1}，再去{WP2}",
    "从{START}出发，先去{WP1}，再到{WP2}，最后到{END}",
    "从{START}先去{WP1}再去{WP2}最后到{END}",
)


MISSING_START_TEMPLATES = (
    "到{END}",
    "去{END}",
    "帮我规划到{END}",
    "终点是{END}",
    "最后到{END}",
    "我想去{END}",
)


MISSING_END_TEMPLATES = (
    "从{START}出发",
    "我要从{START}出发",
    "起点是{START}",
    "从{START}开始",
    "我现在在{START}",
    "我在{START}",
)


MISSING_START_WITH_WAYPOINT_TEMPLATES = (
    "先去{WP1}，再到{WP2}，最后到{END}",
    "先经过{WP1}，再经过{WP2}，最后到{END}",
)


NO_WAYPOINT_WITH_AVOID_TEMPLATES = (
    "从{START}到{END}，避开{AVOID}",
    "从{START}到{END}，不经过{AVOID}",
    "从{START}去{END}，不要经过{AVOID}",
    "我要从{START}到{END}，绕开{AVOID}",
    "帮我规划从{START}到{END}的路线，避开{AVOID}",
    "从{START}到{END}，不要经过{AVOID}",
    "从{START}到{END}，别经过{AVOID}",
    "从{START}到{END}，不路过{AVOID}",
    "从{START}到{END}，不要路过{AVOID}",
    "从{START}到{END}，不走{AVOID}",
    "从{START}到{END}，不要走{AVOID}",
)


ONE_WAYPOINT_WITH_AVOID_TEMPLATES = (
    "从{START}经过{WP}到{END}，避开{AVOID}",
    "从{START}经过{WP}到{END}，不经过{AVOID}",
    "从{START}出发，经过{WP}，最后到{END}，不要经过{AVOID}",
    "从{START}到{END}，途径{WP}，不要经过{AVOID}",
    "从{START}到{END}，先去{WP}，绕开{AVOID}",
)


TWO_WAYPOINT_WITH_AVOID_TEMPLATES = (
    "从{START}出发，先经过{WP1}，再经过{WP2}，最后到{END}，避开{AVOID}",
    "从{START}到{END}，途径{WP1}和{WP2}，不经过{AVOID}",
    "从{START}经过{WP1}再经过{WP2}到{END}，绕开{AVOID}",
)


INVERTED_WITH_AVOID_TEMPLATES = (
    "去{END}，从{START}出发，避开{AVOID}",
    "终点是{END}，起点是{START}，不要经过{AVOID}",
)


MISSING_START_WITH_AVOID_TEMPLATES = (
    "去{END}，避开{AVOID}",
    "到{END}，不经过{AVOID}",
    "帮我规划到{END}，绕开{AVOID}",
)


MISSING_END_WITH_AVOID_TEMPLATES = (
    "从{START}出发，避开{AVOID}",
    "起点是{START}，不要经过{AVOID}",
    "我现在在{START}，绕开{AVOID}",
)


TWO_AVOID_TEMPLATES = (
    "从{START}到{END}，避开{AVOID1}和{AVOID2}",
    "从{START}到{END}，不经过{AVOID1}和{AVOID2}",
)


def generate_bio_training_data(limit: Optional[int] = None) -> List[Dict[str, object]]:
    samples = []
    seen = set()
    for sample in _iter_bio_samples():
        if sample["text"] in seen:
            continue
        validate_bio_sample(sample)
        seen.add(sample["text"])
        samples.append(sample)
    if limit is not None and limit < len(samples):
        return _evenly_sample(samples, limit)
    return samples


def generate_intent_training_data() -> List[Dict[str, str]]:
    samples = [
        {"text": sample["text"], "intent": sample["intent"]}
        for sample in generate_bio_training_data(limit=500)
    ]
    samples.extend(
        [
            {"text": "再经过黄色点", "intent": "add_waypoint"},
            {"text": "帮我加一个青色点作为途经点", "intent": "add_waypoint"},
            {"text": "当前路线怎么走", "intent": "query_route"},
            {"text": "查询当前路径", "intent": "query_route"},
            {"text": "今天天气怎么样", "intent": "unknown"},
            {"text": "讲个笑话", "intent": "unknown"},
            {"text": "打开音乐", "intent": "unknown"},
            {"text": "北京现在几点", "intent": "unknown"},
        ]
    )
    return samples


def validate_bio_sample(sample: Dict[str, object]) -> bool:
    tokens = sample.get("tokens")
    labels = sample.get("labels")
    if not isinstance(tokens, list) or not isinstance(labels, list):
        raise ValueError("Sample must contain tokens and labels lists.")
    if len(tokens) != len(labels):
        raise ValueError("tokens and labels must have the same length.")

    for index, label in enumerate(labels):
        if label not in BIO_TAGS:
            raise ValueError(f"Invalid BIO label: {label}")
        if not label.startswith("I-"):
            continue

        slot = label[2:]
        if index == 0:
            raise ValueError(f"{label} cannot start a sequence.")
        previous = labels[index - 1]
        if previous not in {f"B-{slot}", f"I-{slot}"}:
            raise ValueError(f"{label} cannot follow {previous}.")

    for start, end in _span_ranges(labels):
        if not "".join(tokens[start:end]):
            raise ValueError("BIO span restored to empty text.")

    return True


def _iter_bio_samples() -> Iterable[Dict[str, object]]:
    colors = list(COLOR_ALIASES.keys())
    for start, end in permutations(colors, 2):
        for variant in range(_max_alias_count(start, end)):
            start_text, end_text = _aliases_for_tuple(start, end, variant=variant)
            for template in NO_WAYPOINT_TEMPLATES:
                yield _sample(template, start_text=start_text, end_text=end_text)

            for template in MISSING_START_TEMPLATES:
                yield _sample(template, end_text=end_text)
            for template in MISSING_END_TEMPLATES:
                yield _sample(template, start_text=start_text)

    for start, waypoint, end in permutations(colors, 3):
        for variant in range(_max_alias_count(start, waypoint, end)):
            start_text, waypoint_text, end_text = _aliases_for_tuple(
                start, waypoint, end, variant=variant
            )
            for template in ONE_WAYPOINT_TEMPLATES:
                yield _sample(
                    template,
                    start_text=start_text,
                    waypoint_texts=[waypoint_text],
                    end_text=end_text,
                )

    for start, end, avoid in permutations(colors, 3):
        for variant in range(_max_alias_count(start, end, avoid)):
            start_text, end_text, avoid_text = _aliases_for_tuple(
                start, end, avoid, variant=variant
            )
            for template in NO_WAYPOINT_WITH_AVOID_TEMPLATES:
                yield _sample(
                    template,
                    start_text=start_text,
                    end_text=end_text,
                    avoid_texts=[avoid_text],
                )
            for template in INVERTED_WITH_AVOID_TEMPLATES:
                yield _sample(
                    template,
                    start_text=start_text,
                    end_text=end_text,
                    avoid_texts=[avoid_text],
                )
            for template in MISSING_START_WITH_AVOID_TEMPLATES:
                yield _sample(template, end_text=end_text, avoid_texts=[avoid_text])
            for template in MISSING_END_WITH_AVOID_TEMPLATES:
                yield _sample(template, start_text=start_text, avoid_texts=[avoid_text])

    for start, waypoint, end, avoid in permutations(colors, 4):
        for variant in range(_max_alias_count(start, waypoint, end, avoid)):
            start_text, waypoint_text, end_text, avoid_text = _aliases_for_tuple(
                start, waypoint, end, avoid, variant=variant
            )
            for template in ONE_WAYPOINT_WITH_AVOID_TEMPLATES:
                yield _sample(
                    template,
                    start_text=start_text,
                    waypoint_texts=[waypoint_text],
                    end_text=end_text,
                    avoid_texts=[avoid_text],
                )

    for start, waypoint_a, waypoint_b, end in permutations(colors, 4):
        for variant in range(_max_alias_count(start, waypoint_a, waypoint_b, end)):
            start_text, waypoint_a_text, waypoint_b_text, end_text = _aliases_for_tuple(
                start, waypoint_a, waypoint_b, end, variant=variant
            )
            for template in TWO_WAYPOINT_TEMPLATES:
                yield _sample(
                    template,
                    start_text=start_text,
                    waypoint_texts=[waypoint_a_text, waypoint_b_text],
                    end_text=end_text,
                )
            for template in MISSING_START_WITH_WAYPOINT_TEMPLATES:
                yield _sample(
                    template,
                    waypoint_texts=[waypoint_a_text, waypoint_b_text],
                    end_text=end_text,
                )

    for start, waypoint_a, waypoint_b, end, avoid in permutations(colors, 5):
        for variant in range(_max_alias_count(start, waypoint_a, waypoint_b, end, avoid)):
            start_text, waypoint_a_text, waypoint_b_text, end_text, avoid_text = _aliases_for_tuple(
                start, waypoint_a, waypoint_b, end, avoid, variant=variant
            )
            for template in TWO_WAYPOINT_WITH_AVOID_TEMPLATES:
                yield _sample(
                    template,
                    start_text=start_text,
                    waypoint_texts=[waypoint_a_text, waypoint_b_text],
                    end_text=end_text,
                    avoid_texts=[avoid_text],
                )

    for start, end, avoid_a, avoid_b in permutations(colors, 4):
        for variant in range(_max_alias_count(start, end, avoid_a, avoid_b)):
            start_text, end_text, avoid_a_text, avoid_b_text = _aliases_for_tuple(
                start, end, avoid_a, avoid_b, variant=variant
            )
            for template in TWO_AVOID_TEMPLATES:
                yield _sample(
                    template,
                    start_text=start_text,
                    end_text=end_text,
                    avoid_texts=[avoid_a_text, avoid_b_text],
                )


def _sample(
    template: str,
    start_text: Optional[str] = None,
    waypoint_texts: Optional[Sequence[str]] = None,
    end_text: Optional[str] = None,
    avoid_texts: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    waypoint_texts = list(waypoint_texts or [])
    avoid_texts = list(avoid_texts or [])
    text = template.format(
        START=start_text or "",
        WP=waypoint_texts[0] if waypoint_texts else "",
        WP1=waypoint_texts[0] if waypoint_texts else "",
        WP2=waypoint_texts[1] if len(waypoint_texts) > 1 else "",
        END=end_text or "",
        AVOID=avoid_texts[0] if avoid_texts else "",
        AVOID1=avoid_texts[0] if avoid_texts else "",
        AVOID2=avoid_texts[1] if len(avoid_texts) > 1 else "",
    )
    labels = ["O"] * len(text)
    _tag_span(text, labels, start_text, "START")
    for waypoint_text in waypoint_texts:
        _tag_span(text, labels, waypoint_text, "WAYPOINT")
    _tag_span(text, labels, end_text, "END")
    for avoid_text in avoid_texts:
        _tag_span(text, labels, avoid_text, "AVOID")
    tokens = list(text)
    return {
        "text": text,
        "tokens": tokens,
        "labels": labels,
        "tags": labels,
        "intent": "navigation",
    }


def _tag_span(text: str, labels: List[str], span_text: Optional[str], slot: str) -> None:
    if not span_text:
        return
    start = text.find(span_text)
    if start < 0:
        raise ValueError(f"Span {span_text!r} not found in {text!r}.")
    labels[start] = f"B-{slot}"
    for index in range(start + 1, start + len(span_text)):
        labels[index] = f"I-{slot}"


def _span_ranges(labels: Sequence[str]) -> List[Tuple[int, int]]:
    ranges = []
    start = None
    for index, label in enumerate(labels):
        if label.startswith("B-"):
            if start is not None:
                ranges.append((start, index))
            start = index
        elif label == "O" and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(labels)))
    return ranges


def _aliases_for_tuple(*colors: str, variant: int = 0) -> Tuple[str, ...]:
    return tuple(_alias(color, variant) for color in colors)


def _alias(color: str, offset: int) -> str:
    aliases = COLOR_ALIASES[color]
    return aliases[offset % len(aliases)]


def _max_alias_count(*colors: str) -> int:
    return max(len(COLOR_ALIASES[color]) for color in colors)


def _evenly_sample(samples: List[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    if limit <= 0:
        return []
    if limit == 1:
        return [samples[0]]
    step = (len(samples) - 1) / (limit - 1)
    return [samples[round(index * step)] for index in range(limit)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate path NLP training data.")
    parser.add_argument("--output", default="training_data.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    samples = generate_bio_training_data(limit=args.limit)
    with open(args.output, "w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
