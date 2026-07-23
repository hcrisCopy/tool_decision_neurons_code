# 02 Dataset Preparation

本模块在模型 0/1 标签已经跑完之后使用。

它读取原始 When2Tool 样本和 `$DATA_ROOT/labels/<model_alias>/`，输出每个模型自己的改造后数据集：原始字段保留，同时加入 A/B/C 任务类型和 `tool_necessary` 标签。

运行：

```bash
bash scripts/run_02_build_modified_when2tool.sh
```

等价 Python 命令：

```bash
python code/02_dataset_preparation/build_modified_when2tool.py \
  --data-root ../tool_decision_neurons_data \
  --model-aliases qwen3-1.7b \
  --overwrite
```

输入：

```text
../tool_decision_neurons_data/datasets/raw_when2tool/
../tool_decision_neurons_data/labels/<model_alias>/
```

输出：

```text
../tool_decision_neurons_data/datasets/modified_when2tool/<model_alias>/
```

说明：这个阶段不重新跑模型，只合并已经生成的标签；默认要求标签覆盖对应 split 的全部原始样本。
