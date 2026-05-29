# Hybrid NLP Parser for Color-Based Path Planning

面向七色点路径规划任务的混合式自然语言理解模块。

## 1. 项目简介

本项目实现一个用于地图路径规划任务的 NLP 模块。用户输入中文路径描述，系统输出结构化 JSON，供后端 CV/MDP 路径规划模块使用。

NLP 模块负责：

- 识别用户是否在表达路径规划需求。
- 抽取起点 `start`、途经点 `waypoints`、终点 `end`。
- 抽取避让点 `avoid_points`，用于表达“不经过/避开/绕开”等约束。
- 将中文颜色别名归一化为英文颜色 key。
- 在模型抽取失败时使用规则解析器兜底。

NLP 模块不负责：

- 输出地图坐标。
- 执行路径搜索。
- 执行 MDP 策略求解。
- 执行 CV 地图识别。

示例输入：

```text
我要从蓝色点出发，经过紫，到蓝色，不要经过黄色
```

示例输出：

```json
{
  "intent": "navigation",
  "start": "blue",
  "waypoints": ["purple"],
  "end": "blue",
  "avoid_points": ["yellow"],
  "is_complete": true,
  "missing_slots": []
}
```

说明：黄色出现在“不经过黄色”中，因此不会加入 `waypoints`，而会进入 `avoid_points`。

## 2. 支持的颜色点

| 中文颜色 | 标准 key | 支持别名 |
|---|---|---|
| 红/赤 | `red` | 红、红点、红色、红色点、赤、赤点、赤色、赤色点 |
| 橙 | `orange` | 橙、橙点、橙色、橙色点 |
| 黄 | `yellow` | 黄、黄点、黄色、黄色点 |
| 绿 | `green` | 绿、绿点、绿色、绿色点 |
| 青 | `cyan` | 青、青点、青色、青色点 |
| 蓝 | `blue` | 蓝、蓝点、蓝色、蓝色点 |
| 紫 | `purple` | 紫、紫点、紫色、紫色点 |

## 3. 系统架构

整体流程：

```text
用户输入
  -> IntentClassifier
  -> BiLSTM-CRF SlotTagger
  -> ColorNormalizer
  -> HybridPathNLP
  -> PathNLPParser fallback
  -> 固定 JSON 输出
```

核心组件：

| 组件 | 说明 |
|---|---|
| `IntentClassifier` | 使用 `TfidfVectorizer + LogisticRegression` 判断 `navigation` / `unknown` 等意图。 |
| `BiLSTMCRFSlotTagger` | 使用 PyTorch 字符级 BiLSTM-CRF 做 BIO 序列标注，抽取 `START` / `END` / `WAYPOINT` / `AVOID`。 |
| `ColorNormalizer` | 将“蓝色点 / 蓝点 / 蓝”等中文别名归一化为 `blue` 等标准 key。 |
| `PathNLPParser` | 规则解析器，作为 fallback，提高系统鲁棒性。 |
| `HybridPathNLP` | 对外统一入口，模型优先，失败时规则兜底。 |

## 4. BIO 标签说明

训练数据使用字符级 BIO 标注。

示例：

```json
{
  "text": "从蓝色点到绿色点",
  "tokens": ["从", "蓝", "色", "点", "到", "绿", "色", "点"],
  "labels": ["O", "B-START", "I-START", "I-START", "O", "B-END", "I-END", "I-END"]
}
```

标签集合：

- `O`
- `B-START` / `I-START`
- `B-END` / `I-END`
- `B-WAYPOINT` / `I-WAYPOINT`
- `B-AVOID` / `I-AVOID`

“不经过/避开/绕开”后的颜色在 BIO 中标为 `AVOID`，最终输出到 `avoid_points`，不作为 waypoint。

## 5. 安装环境

推荐 Python 版本：

```text
Python 3.9+
```

安装依赖：

```bash
pip install -r requirements.txt
```

依赖包括：

- `torch`
- `scikit-learn`

`tkinter` 是 Python 标准库，不写入 `requirements.txt`。

## 6. 训练模型

训练并保存 slot tagger：

```bash
python train_slot_tagger.py --epochs 30 --output slot_tagger.pkl
```

小样本冒烟测试：

```bash
python train_slot_tagger.py --limit 5 --epochs 1 --output slot_tagger_test.pkl
```

如果当前目录存在 `slot_tagger.pkl`，`HybridPathNLP` 会自动加载。  
如果不存在，系统会进入 fallback-heavy mode，但仍可通过规则 parser 保持基本解析能力。

`slot_tagger.pkl` 是本地训练产物，默认被 `.gitignore` 忽略，不建议作为大模型文件提交。

## 7. 自动测试

运行：

```bash
python -m unittest
```

预期：所有测试通过。

## 8. 自动评估

运行：

```bash
python evaluate_hybrid_nlp.py
```

最近一次评估结果：

```text
Total samples: 112
Intent accuracy: 1.0000
Start accuracy: 1.0000
End accuracy: 1.0000
Waypoint exact match accuracy: 1.0000
Avoid exact match accuracy: 1.0000
Semantic frame accuracy: 1.0000
Fallback used: 38 / 112
Fallback rate: 0.3393
No prediction errors.
```

指标解释：

- `Intent accuracy`：意图识别准确率。
- `Start accuracy`：起点识别准确率。
- `End accuracy`：终点识别准确率。
- `Waypoint exact match accuracy`：途经点列表完全匹配率，包括顺序。
- `Avoid exact match accuracy`：避让点列表完全匹配率，包括顺序。
- `Semantic frame accuracy`：`start`、`waypoints`、`end`、`avoid_points` 全部正确才算正确。
- `Fallback rate`：使用规则兜底的比例。

## 9. 人工测试方式

命令行测试：

```bash
python interactive_cli.py
```

GUI 测试：

```bash
python interactive_gui.py
```

GUI 会显示：

- `result`
- `debug`
- `model_loaded`
- `slot_source`
- `used_fallback`
- `fallback_reason`

推荐人工测试句子：

1. 从蓝色点到绿色点
2. 蓝到绿
3. 去绿色点，从蓝色点出发
4. 从蓝色点出发，经过青色点，最后到绿色点
5. 从蓝到红，途径黄和紫
6. 我要从蓝色点出发，经过紫，到蓝色，不要经过黄色
7. 从红点到蓝点，途径橙点，避开黄色点
8. 从蓝到红，避开黄和紫
9. 去绿色点，避开红点
10. 从蓝色点出发，不经过黄色点
11. 从青色点到蓝色点
12. 到绿色点
13. 从蓝色点出发
14. 今天天气怎么样

## 10. Python API 用法

正式接口：

```python
from hybrid_path_nlp import HybridPathNLP

parser = HybridPathNLP()
result = parser.parse("从蓝色点出发，经过青色点，最后到绿色点")
print(result)
```

输出：

```json
{
  "intent": "navigation",
  "start": "blue",
  "waypoints": ["cyan"],
  "end": "green",
  "avoid_points": [],
  "is_complete": true,
  "missing_slots": []
}
```

调试接口：

```python
debug_result = parser.parse_with_debug("从蓝色点到绿色点")
print(debug_result)
```

`parse(text)` 是正式接口。  
`parse_with_debug(text)` 只用于调试模型输出和 fallback 行为，不建议后端直接使用。

## 11. 与后端接口约定

后端路径规划模块只需要读取：

- `start`
- `waypoints`
- `end`
- `avoid_points`

例如：

```json
{
  "start": "blue",
  "waypoints": ["cyan", "purple"],
  "end": "green",
  "avoid_points": ["yellow"]
}
```

如果后端暂时不支持 `avoid_points`，可以忽略该字段；NLP 只负责识别颜色 key，不判断地图上是否真的可避开。

NLP 不负责：

- 地图坐标。
- 障碍物。
- 路径搜索。
- MDP 策略求解。
- CV 地图识别。

## 12. 当前限制与未来扩展

当前限制：

1. 当前只支持七个颜色点。
2. NLP 识别 `avoid_points`，但不判断具体地图路径是否能避开。
3. 具体避让路径由后端 CV/MDP 模块处理。
4. 当前训练数据主要由模板自动生成，适合本课程任务，但不代表开放域泛化能力。
5. 如果未来需要更复杂约束，可以继续扩展槽位，例如禁止边、偏好点或动态障碍。

## 13. 项目文件说明

| 文件 | 作用 |
|---|---|
| `hybrid_path_nlp.py` | 总入口，模型优先，规则兜底 |
| `path_nlp_parser.py` | 规则解析器 fallback |
| `slot_tagger.py` | BiLSTM-CRF 槽位标注模型 |
| `intent_classifier.py` | TF-IDF + LogisticRegression 意图分类 |
| `color_normalizer.py` | 颜色别名归一化 |
| `generate_training_data.py` | 自动生成 BIO 训练数据 |
| `train_slot_tagger.py` | 训练并保存 slot tagger |
| `evaluate_hybrid_nlp.py` | 自动评估脚本 |
| `interactive_cli.py` | 命令行人工测试 |
| `interactive_gui.py` | GUI 人工测试窗口 |
| `test_hybrid_nlp.py` | Hybrid NLP 单元测试 |
| `test_path_nlp_parser.py` | 规则 parser 单元测试 |
| `README_NLP.md` | NLP 模块专项技术说明 |
| `requirements.txt` | Python 依赖列表 |

## 14. 课程知识点

本项目涉及的 NLP 和工程知识点包括：

- Task-oriented Dialogue System
- Intent Classification
- TF-IDF
- Logistic Regression
- Slot Filling
- BIO Sequence Labeling
- BiLSTM
- CRF
- Entity Normalization
- Hybrid NLP Pipeline
- Fallback Mechanism
- Semantic Frame Evaluation
