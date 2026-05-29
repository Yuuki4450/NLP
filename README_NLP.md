# NLP Module

## Goal

This module parses Chinese map path-planning requests into structured JSON using color keys. It is designed for a map with seven color points:

`red`, `orange`, `yellow`, `green`, `cyan`, `blue`, `purple`

The NLP layer only extracts the requested route frame. It does not output coordinates, does not search for paths, and does not run CV or MDP logic. Backend CV/MDP modules should use `start`, `waypoints`, `end`, and optionally `avoid_points` color keys to perform path planning.

## Interface

Use `HybridPathNLP.parse(text)` for normal inference.

Input:

```text
从蓝色点出发，经过青色点和紫色点，最后到绿色点
```

Output:

```json
{
  "intent": "navigation",
  "start": "blue",
  "waypoints": ["cyan", "purple"],
  "end": "green",
  "avoid_points": [],
  "is_complete": true,
  "missing_slots": []
}
```

The final `parse(text)` output fields are fixed:

```json
{
  "intent": "navigation or unknown",
  "start": "color key or null",
  "waypoints": ["color key"],
  "end": "color key or null",
  "avoid_points": ["color key"],
  "is_complete": true,
  "missing_slots": []
}
```

`waypoints` are required intermediate points. `avoid_points` are colors that the user asks the backend to avoid, for example after `不经过`, `避开`, or `绕开`.

For diagnostics, use `HybridPathNLP.parse_with_debug(text)`. It wraps the same parse result with fallback metadata:

```json
{
  "result": {
    "intent": "navigation",
    "start": "blue",
    "waypoints": [],
    "end": "green",
    "avoid_points": [],
    "is_complete": true,
    "missing_slots": []
  },
  "used_fallback": true,
  "fallback_reason": "missing_required_slots",
  "slot_source": "fallback"
}
```

## Color Keys

| Chinese | English key |
| --- | --- |
| 红/赤 | `red` |
| 橙 | `orange` |
| 黄 | `yellow` |
| 绿 | `green` |
| 青 | `cyan` |
| 蓝 | `blue` |
| 紫 | `purple` |

## Architecture

- `IntentClassifier`: TF-IDF character features with `LogisticRegression`.
- `BiLSTMCRFSlotTagger`: PyTorch character-level BiLSTM-CRF BIO sequence tagger for `START`, `END`, `WAYPOINT`, and `AVOID`.
- `ColorNormalizer`: maps aliases such as `蓝色点`, `蓝点`, and `蓝` to `blue`.
- `PathNLPParser`: rule-based parser used as fallback.
- `HybridPathNLP`: hybrid model-first parser with rule fallback. It uses the model result when colors can be normalized and the slot frame is valid. It falls back only when the slot tagger fails, returns invalid output, returns no useful slots, or produces color text that cannot be normalized.

The system is intentionally hybrid:

- Model first: `IntentClassifier` and `BiLSTMCRFSlotTagger` handle normal inference.
- Rule fallback: `PathNLPParser` preserves stability for malformed or low-confidence slot outputs.

## BIO Labels

The slot tagger uses character-level BIO labels:

- `O`
- `B-START` / `I-START`
- `B-END` / `I-END`
- `B-WAYPOINT` / `I-WAYPOINT`
- `B-AVOID` / `I-AVOID`

Training examples mark colors after `不经过`, `不要经过`, `避开`, `绕开`, `不走`, and similar phrases as `AVOID`, not `WAYPOINT`.

## Run Tests

```bash
python -m unittest
```

## Train Slot Tagger

```bash
python train_slot_tagger.py --epochs 30 --output slot_tagger.pkl
```

Small smoke test:

```bash
python train_slot_tagger.py --limit 5 --epochs 1 --output slot_tagger_test.pkl
```

## Generate Training Data

```bash
python generate_training_data.py --output training_data.jsonl
```

## Run Evaluation

```bash
python evaluate_hybrid_nlp.py
```

Recommended workflow:

```bash
python train_slot_tagger.py --epochs 30 --output slot_tagger.pkl
python evaluate_hybrid_nlp.py
```

If `slot_tagger.pkl` is not present, `evaluate_hybrid_nlp.py` builds an in-memory slot tagger from generated BIO samples for evaluation.

The report includes:

- `semantic_frame_accuracy`: `start`, `waypoints`, `end`, and `avoid_points` must all match.
- `waypoint_exact_match_accuracy`: waypoint list must match exactly, including order.
- `avoid_exact_match_accuracy`: avoid list must match exactly, including order.
- `fallback_rate`: fraction of samples resolved by the rule fallback. Lower means the BiLSTM-CRF slot model is handling more cases independently. Higher means the rule parser is still carrying more of the stability burden.
- `fallback_reason_counts`: distribution of fallback causes, such as normalization failures or invalid slot output.

Use verbose mode to inspect fallback examples:

```bash
python evaluate_hybrid_nlp.py --verbose
```

## Manual Testing

Command-line interactive test:

```bash
python interactive_cli.py
```

GUI window test:

```bash
python interactive_gui.py
```

Before running the CLI or GUI with the trained slot model, train and save the model in the project directory:

```bash
python train_slot_tagger.py --epochs 30 --output slot_tagger.pkl
```

If `slot_tagger.pkl` is present, CLI and GUI show `loaded slot tagger from slot_tagger.pkl`. If it is missing, they show `slot_tagger.pkl not found, using fallback-heavy mode`, and `parse_with_debug` includes `model_loaded: false`.

Automated evaluation:

```bash
python evaluate_hybrid_nlp.py
```

Unit tests:

```bash
python -m unittest
```

Recommended manual test sentences:

- 从蓝色点到绿色点
- 从蓝色点到绿色点，不经过紫色点
- 我要从蓝色点出发，经过紫，到蓝色，不要经过黄色
- 从红点到蓝点，途径橙点，避开黄色点
- 从青点到紫点，先经过绿点，不要路过橙点
- 从蓝到红，避开黄和紫
- 去绿色点，避开红点
- 从蓝色点出发，不经过黄色点
- 蓝到绿
- 去绿色点，从蓝色点出发
- 从蓝色点出发，经过青色点，最后到绿色点
- 从蓝到红，途径黄和紫
- 从红点先去橙点再去黄点最后到蓝点
- 从青色点到蓝色点
- 从蓝色点到青色点
- 到绿色点
- 从蓝色点出发
- 今天天气怎么样

`parse(text)` returns only the stable backend-facing fields. `parse_with_debug(text)` is for local debugging of model output and fallback behavior. The CLI and GUI tools are local test aids only; they do not change the backend interface.

Avoid constraints such as `不经过紫色点`, `不要经过黄色`, `避开黄色点`, and `绕开黄` are normalized into `avoid_points`. They are not added to `waypoints`. If the backend does not support avoid constraints yet, it can ignore `avoid_points`.

## Scope

This NLP module does not output coordinates. It does not perform path search. It only outputs `start`, `waypoints`, `end`, and `avoid_points` as color keys. The backend CV/MDP path-planning module is responsible for using those color keys to compute actual movement plans and decide whether an avoid constraint is feasible.
