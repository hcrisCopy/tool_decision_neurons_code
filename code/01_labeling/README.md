# 01 Labeling

本阶段使用 When2Tool 原始数据集生成每个模型自己的 `tool_necessary=0/1` 标签。

核心逻辑对齐 When2Tool Step 2：

```text
prompt_mode = hard_no_tool
reasoning_mode = no_reasoning
tool_necessary = 0 if model answers correctly without tools else 1
```

输入：

```text
../tool_decision_neurons_data/datasets/raw_when2tool/
```

输出：

```text
../tool_decision_neurons_data/labels/<model_alias>/
```

单模型 demo：

```bash
bash scripts/run_01_labeling_demo.sh qwen3-1.7b
```

全模型脚本：

```bash
bash scripts/run_01_labeling_all.sh
```
