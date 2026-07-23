# Cross-Task Tool-Decision Neurons

一句话简介：本项目研究大模型中是否存在跨 When2Tool A/B/C 任务类型共享的“是否调工具”关键神经元，并按 Who Transfers Safety? 方法完成探测、因果验证和训练。

## To-do List

- [ ] 搭建 GitHub 代码仓库目录
- [ ] 准备原始 When2Tool 数据集
- [ ] 构建改造后的 When2Tool 数据集
- [ ] 下载目标模型到项目同级目录
- [ ] 为所有目标模型分别生成 `tool_necessary=0/1` 标签
- [ ] 提取模型隐藏状态、激活和干预所需中间结果
- [ ] 探测 A/B/C 单类型工具决策神经元
- [ ] 做单类型因果验证
- [ ] 取 A/B/C 交集得到跨类型共享神经元
- [ ] 做跨类型因果验证
- [ ] 训练跨类型共享工具决策神经元
- [ ] 对齐 When2Tool 指标完成最终评测和结果汇总

## 章节目录

- [目录结构](#目录结构)
- [数据和模型资源](#数据和模型资源)
- [阶段 0：环境配置](#阶段-0环境配置)
- [阶段 1：原始数据准备](#阶段-1原始数据准备)
- [阶段 2：改造后数据集构建](#阶段-2改造后数据集构建)
- [阶段 3：标签生成](#阶段-3标签生成)
- [阶段 4：特征和激活提取](#阶段-4特征和激活提取)
- [阶段 5：单类型神经元探测](#阶段-5单类型神经元探测)
- [阶段 6：单类型因果验证](#阶段-6单类型因果验证)
- [阶段 7：共享神经元发现](#阶段-7共享神经元发现)
- [阶段 8：跨类型因果验证](#阶段-8跨类型因果验证)
- [阶段 9：神经元训练](#阶段-9神经元训练)
- [阶段 10：评测和汇总](#阶段-10评测和汇总)
- [命名规范](#命名规范)
- [参考链接](#参考链接)

## 目录结构

本 README 位于 `tool_decision_neurons_code/` 根目录。这个目录就是 GitHub 仓库，只放代码、配置、脚本和 README。数据、模型权重、训练 ckpt、输出结果和日志都不进 GitHub，在后面对应阶段单独说明。

代码仓库结构：

```text
tool_decision_neurons_code/
|-- README.md                              # 项目总说明，GitHub 首页入口
|-- .gitignore                             # 排除缓存、日志、数据、ckpt 等大文件
|-- requirements.txt                       # Python 依赖；torch 单独安装
|-- configs/                               # 后续各阶段配置文件
|   |-- paths.yaml                         # DATA_ROOT / MODEL_ROOT 等路径配置
|   |-- models.yaml                        # 目标模型 repo_id、alias、推理参数
|   |-- data.yaml                          # 数据集 split 和字段配置
|   |-- labeling.yaml                      # 跑 tool_necessary 标签配置
|   |-- probing.yaml                       # 神经元探测配置
|   |-- causal_single.yaml                 # 单类型因果验证配置
|   |-- causal_cross.yaml                  # 跨类型因果验证配置
|   |-- training.yaml                      # 神经元训练配置
|   |-- evaluation.yaml                    # When2Tool 对齐评测配置
|-- code/
|   |-- common/                            # 公共工具代码
|   |   |-- env_type_mapping.py            # When2Tool env_name 到 A/B/C 的映射
|   |   |-- model_registry.py              # 后续补：模型 alias 和路径管理
|   |   |-- paths.py                       # 后续补：统一路径解析
|   |   |-- metrics.py                     # 后续补：When2Tool 指标
|   |   |-- io_utils.py                    # 后续补：JSONL/parquet 读写
|   |-- 00_dataset_preparation/            # 阶段 2：改造原始数据集，只加类型映射
|   |   |-- build_modified_when2tool.py    # 生成 modified_when2tool
|   |   |-- README.md                      # 本阶段说明
|   |-- 01_labeling/                       # 阶段 3：每个模型跑 tool_necessary 标签
|   |-- 02_feature_extraction/             # 阶段 4：提取隐藏状态和激活
|   |-- 03_single_type_neuron_probing/     # 阶段 5：A/B/C 单类型神经元探测
|   |-- 04_single_type_causal_validation/  # 阶段 6：单类型因果验证
|   |-- 05_shared_neuron_discovery/        # 阶段 7：A/B/C 交集共享神经元
|   |-- 06_cross_type_causal_validation/   # 阶段 8：跨类型因果验证
|   |-- 07_training/                       # 阶段 9：神经元训练
|   |-- 08_evaluation/                     # 阶段 10：指标汇总和评测
|   |-- third_party/                       # 第三方代码适配，不直接混入主逻辑
|       |-- when2tool_adapter/             # When2Tool 原方法适配
|       |-- ss_neuron_expansion_adapter/   # Who Transfers Safety? 方法适配
|-- scripts/
    |-- run_00_build_modified_when2tool.sh # 一键生成改造后数据集
    |-- run_01_labeling.sh                 # 后续补：一键跑标签
    |-- run_02_feature_extraction.sh       # 后续补：一键提特征
    |-- run_03_single_type_neuron_probing.sh
    |-- run_04_single_type_causal_validation.sh
    |-- run_05_shared_neuron_discovery.sh
    |-- run_06_cross_type_causal_validation.sh
    |-- run_07_training.sh
    |-- run_08_evaluation.sh
```

默认路径约定：

```text
CODE_ROOT=.
DATA_ROOT=../tool_decision_neurons_data
MODEL_ROOT=..
```

说明：

- GitHub 只同步 `tool_decision_neurons_code/`。
- 代码里不要硬编码个人电脑路径，所有数据和模型位置从配置文件或命令行参数传入。
- `tool_decision_neurons_data/` 和模型权重目录与代码仓库同级，不提交 GitHub。
- 每个模型的实验输出必须放在自己的 `<model_alias>/` 目录下，避免不同模型互相覆盖。

## 数据和模型资源

### 数据集

| 名称 | 类型 | 来源 | 下载或交接地址 | 推荐放置路径 | 用途 |
|---|---|---|---|---|---|
| When2Tool 原始数据 | dataset | Hugging Face | https://huggingface.co/datasets/cesun/When2Tool | `$DATA_ROOT/datasets/raw_when2tool/` | 原始 single-hop / multi-hop 数据 |
| When2Tool 代码 | code | GitHub | https://github.com/Trustworthy-ML-Lab/when2tool | `code/third_party/when2tool_adapter/` | 标签生成和指标对齐参考 |
| 改造后 When2Tool 数据 | processed dataset | 由原始 When2Tool 生成 | 百度网盘链接待补充 | `$DATA_ROOT/datasets/modified_when2tool/` | 按 env 映射 A/B/C 后的实验输入 |
| 模型标签结果 | generated labels | 本项目生成 | 随实验输出交接 | `$DATA_ROOT/labels/<model_alias>/` | 每个模型自己的 `tool_necessary=0/1` 标签 |

### 模型

模型权重不放 GitHub，也不放进代码仓库。直接按 Hugging Face repo_id 的两级目录放在代码仓库同级位置，例如 `Qwen/Qwen3-4B-Instruct-2507` 对应 `$MODEL_ROOT/Qwen/Qwen3-4B-Instruct-2507/`。

| 模型 alias | Hugging Face repo_id | 下载地址 | 推荐放置路径 | 备注 |
|---|---|---|---|---|
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | https://huggingface.co/Qwen/Qwen3-1.7B | `$MODEL_ROOT/Qwen/Qwen3-1.7B/` | Qwen3 dense model |
| `qwen3-4b-instruct` | `Qwen/Qwen3-4B-Instruct-2507` | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 | `$MODEL_ROOT/Qwen/Qwen3-4B-Instruct-2507/` | instruct model |
| `qwen3-14b` | `Qwen/Qwen3-14B` | https://huggingface.co/Qwen/Qwen3-14B | `$MODEL_ROOT/Qwen/Qwen3-14B/` | Qwen3 dense model |
| `qwen3-32b` | `Qwen/Qwen3-32B` | https://huggingface.co/Qwen/Qwen3-32B | `$MODEL_ROOT/Qwen/Qwen3-32B/` | Qwen3 dense model |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | `$MODEL_ROOT/meta-llama/Llama-3.1-8B-Instruct/` | 需要确认 HF 访问权限 |
| `llama3.3-70b` | `meta-llama/Llama-3.3-70B-Instruct` | https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct | `$MODEL_ROOT/meta-llama/Llama-3.3-70B-Instruct/` | 需要确认 HF 访问权限 |

## 阶段 0：环境配置

建议使用 Python 3.10。PyTorch 单独安装，其他依赖从 `requirements.txt` 安装。

```bash
conda create -n tool_neurons python=3.10 -y
conda activate tool_neurons

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

如果当前机器暂时不支持 vLLM，可以先把 `requirements.txt` 里的 `vllm==0.8.5` 注释掉；正式跑多模型标签，尤其是 70B 模型时，建议在 Linux GPU 服务器上安装 vLLM。

安装后简单检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers, datasets, vllm; print(transformers.__version__)"
```

## 阶段 1：原始数据准备

When2Tool 原始数据放在 `$DATA_ROOT/datasets/raw_when2tool/`，不提交到 GitHub。本阶段只确认原始数据存在，不改字段、不生成标签。

```text
$DATA_ROOT/
|-- datasets/
    |-- raw_when2tool/
        |-- single_hop/
        |   |-- train-*.parquet
        |   |-- test-*.parquet
        |-- multi_hop/
            |-- train-*.parquet
            |-- test-*.parquet
```

When2Tool 原始数据划分：

| subset | split | 数量 | 原始文件位置 | 说明 |
|---|---|---:|---|---|
| `single_hop` | train | 900 | `$DATA_ROOT/datasets/raw_when2tool/single_hop/train-*.parquet` | 单跳训练集 |
| `single_hop` | test | 2250 | `$DATA_ROOT/datasets/raw_when2tool/single_hop/test-*.parquet` | 单跳测试集 |
| `multi_hop` | train | 180 | `$DATA_ROOT/datasets/raw_when2tool/multi_hop/train-*.parquet` | 多跳训练集 |
| `multi_hop` | test | 450 | `$DATA_ROOT/datasets/raw_when2tool/multi_hop/test-*.parquet` | 多跳测试集 |

原始字段：

```text
id, difficulty, multi_step, instruction, env_name, tools, parameters, answer, steps, tags
```

## 阶段 2：改造后数据集构建

本阶段只做种类映射：根据 `env_name` 给每条样本加上 A/B/C 任务类型。不会生成 `tool_necessary`，不会修改 `instruction/tools/answer`，不会删除样本，也不会做模型相关处理。

运行命令：

```bash
conda activate tool_neurons
bash scripts/run_00_build_modified_when2tool.sh
```

A/B/C 映射：

| task_type | When2Tool 类别 | env_name |
|---|---|---|
| `A` | Computational Scale | `CalculatorEnv`, `StatisticsEnv`, `CountingEnv`, `MatrixEnv`, `PrimeEnv` |
| `B` | Knowledge Boundaries | `RetrieverEnv`, `HistoricalYearEnv`, `GameRuleEnv`, `HashEnv`, `DecodingEnv` |
| `C` | Execution Reliability | `ListManipulationEnv`, `DateTimeEnv`, `CodeExecutorEnv`, `ScheduleEnv`, `RegexMatchEnv` |

改造后样本新增字段：

| 字段 | 含义 | 是否来自 When2Tool 原始字段 |
|---|---|---|
| `source_dataset` | 数据来源，固定为 `When2Tool` | 否，本项目新增 |
| `subset` | `single_hop` 或 `multi_hop` | 否，本项目新增 |
| `split` | `train` 或 `test` | 否，本项目新增 |
| `sample_uid` | 全局唯一样本 id，格式为 `subset:split:original_id` | 否，本项目新增 |
| `original_id` | 原始 When2Tool 的 `id` 备份 | 否，本项目新增 |
| `task_type` | A/B/C 任务类型 | 否，本项目新增 |
| `task_type_name` | A/B/C 的英文类别名 | 否，本项目新增 |
| `when2tool_category` | When2Tool 原论文中的类别名 | 否，本项目新增 |

阶段二输出：

| 输出 | 位置 | 作用 |
|---|---|---|
| 改造后 single-hop train | `$DATA_ROOT/datasets/modified_when2tool/single_hop/train.jsonl` 和 `.parquet` | 保留原始样本，额外加 A/B/C 映射字段 |
| 改造后 single-hop test | `$DATA_ROOT/datasets/modified_when2tool/single_hop/test.jsonl` 和 `.parquet` | 后续标签生成、探测、验证使用 |
| 改造后 multi-hop train | `$DATA_ROOT/datasets/modified_when2tool/multi_hop/train.jsonl` 和 `.parquet` | 多跳训练 split |
| 改造后 multi-hop test | `$DATA_ROOT/datasets/modified_when2tool/multi_hop/test.jsonl` 和 `.parquet` | 多跳测试 split |
| 映射文件 | `$DATA_ROOT/datasets/modified_when2tool/env_type_mapping.json` | 记录 `env_name -> A/B/C` |
| 汇总表 | `$DATA_ROOT/datasets/modified_when2tool/summary.csv` | 按 subset/split/task_type/env/difficulty 统计数量 |
| manifest | `$DATA_ROOT/datasets/modified_when2tool/manifest.json` | 记录输入输出路径、字段和样本数量 |
| 数据说明 | `$DATA_ROOT/datasets/modified_when2tool/README.md` | 改造后数据集说明 |
| 网盘交接占位 | `$DATA_ROOT/datasets/modified_when2tool/baidu_netdisk_info.md` | 后续补百度网盘链接和提取码 |

说明：之前生成过的 `by_type/` 目录只是按 `task_type` 复制出来的快捷分组，不是新数据。现在脚本已经取消生成 `by_type/`，后续如果要拿 A/B/C 数据，直接读取 train/test 后按 `task_type` 字段筛选即可。

## 阶段 3：标签生成

待补充。

需要覆盖的模型：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```

标签输出放在 `$DATA_ROOT/labels/`。

```text
$DATA_ROOT/
|-- labels/
    |-- <model_alias>/
        |-- single_hop/
        |-- multi_hop/
```

## 阶段 4：特征和激活提取

待补充。

特征和激活输出放在 `$DATA_ROOT/features/`。

```text
$DATA_ROOT/
|-- features/
    |-- <model_alias>/
        |-- single_hop/
        |-- multi_hop/
```

## 阶段 5：单类型神经元探测

待补充。

A/B/C 单类型神经元结果放在 `$DATA_ROOT/neurons/<model_alias>/single_type/`。

```text
$DATA_ROOT/
|-- neurons/
    |-- <model_alias>/
        |-- single_type/
            |-- A/
            |-- B/
            |-- C/
```

## 阶段 6：单类型因果验证

待补充。

单类型因果验证结果放在 `$DATA_ROOT/causal_validation/<model_alias>/single_type/`。

```text
$DATA_ROOT/
|-- causal_validation/
    |-- <model_alias>/
        |-- single_type/
```

## 阶段 7：共享神经元发现

待补充。

A/B/C 交集得到的跨类型共享神经元放在 `$DATA_ROOT/neurons/<model_alias>/shared/`。

```text
$DATA_ROOT/
|-- neurons/
    |-- <model_alias>/
        |-- shared/
```

## 阶段 8：跨类型因果验证

待补充。

跨类型因果验证结果放在 `$DATA_ROOT/causal_validation/<model_alias>/cross_type/`。

```text
$DATA_ROOT/
|-- causal_validation/
    |-- <model_alias>/
        |-- cross_type/
```

## 阶段 9：神经元训练

待补充。

训练数据和训练 ckpt 放在 `$DATA_ROOT/training_data/` 与 `$DATA_ROOT/checkpoints/`。

```text
$DATA_ROOT/
|-- training_data/
|   |-- <model_alias>/
|-- checkpoints/
    |-- <model_alias>/
```

## 阶段 10：评测和汇总

待补充。

最终评测输出、日志和图表分别放在 `$DATA_ROOT/outputs/`、`$DATA_ROOT/logs/`、`$DATA_ROOT/visualizations/`。

```text
$DATA_ROOT/
|-- outputs/
|   |-- <model_alias>/
|-- logs/
|   |-- <model_alias>/
|-- visualizations/
    |-- <model_alias>/
```

## 命名规范

模型 alias 固定使用：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```

代码阶段目录固定使用：

```text
00_dataset_preparation
01_labeling
02_feature_extraction
03_single_type_neuron_probing
04_single_type_causal_validation
05_shared_neuron_discovery
06_cross_type_causal_validation
07_training
08_evaluation
```

每个阶段输出建议包含：

```text
config.yaml
metrics.json
summary.md
run.log
manifest.json
```

## 参考链接

- Who Transfers Safety?: https://arxiv.org/abs/2602.01283
- LLM Agents Already Know When to Call Tools: https://arxiv.org/abs/2605.09252
- When2Tool dataset: https://huggingface.co/datasets/cesun/When2Tool
- When2Tool code: https://github.com/Trustworthy-ML-Lab/when2tool
- SS-Neuron-Expansion code: https://github.com/1518630367/SS-Neuron-Expansion
