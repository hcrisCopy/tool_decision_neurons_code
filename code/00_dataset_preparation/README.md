# 00 Dataset Preparation

本模块只做一件事：把 When2Tool 原始样本按 `env_name` 映射到 A/B/C 三类任务。

运行：

```bash
bash scripts/run_00_build_modified_when2tool.sh
```

等价 Python 命令：

```bash
python code/00_dataset_preparation/build_modified_when2tool.py \
  --data-root ../tool_decision_neurons_data \
  --overwrite
```

输入：

```text
../tool_decision_neurons_data/datasets/raw_when2tool/
```

输出：

```text
../tool_decision_neurons_data/datasets/modified_when2tool/
```

说明：这个阶段不生成 `tool_necessary`，不修改 `instruction/tools/answer`，不筛样本，只额外写入任务类型字段。
