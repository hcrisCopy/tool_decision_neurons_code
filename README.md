# Cross-Task Tool-Decision Neurons

一句话简介：本项目研究大模型中是否存在跨 When2Tool A/B/C 任务类型共享的“是否调工具”关键神经元，并按 Who Transfers Safety? 方法完成探测、因果验证和训练。

## To-do List

- [ ] 搭建 GitHub 代码仓库目录
- [ ] 准备原始 When2Tool 数据集
- [ ] 下载目标模型到项目同级目录
- [ ] 为所有目标模型分别生成 `tool_necessary=0/1` 标签
- [ ] 基于标签构建改造后的 When2Tool 数据集
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
- [阶段 2：标签生成](#阶段-2标签生成)
- [阶段 3：改造后数据集构建](#阶段-3改造后数据集构建)
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
|   |-- 01_labeling/                       # 阶段 2：每个模型跑 tool_necessary 标签
|   |   |-- build_when2tool_labels.py      # 对齐 When2Tool hard_no_tool 生成 0/1 标签
|   |   |-- README.md                      # 本阶段说明
|   |-- 02_dataset_preparation/            # 阶段 3：基于标签构建改造后数据集
|   |   |-- build_modified_when2tool.py    # 合并 raw 样本、A/B/C 映射和 0/1 标签
|   |   |-- README.md                      # 本阶段说明
|   |-- 03_feature_extraction/             # 阶段 4：提取隐藏状态和激活
|   |-- 04_single_type_neuron_probing/     # 阶段 5：A/B/C 单类型神经元探测
|   |-- 05_single_type_causal_validation/  # 阶段 6：单类型因果验证
|   |-- 06_shared_neuron_discovery/        # 阶段 7：A/B/C 交集共享神经元
|   |-- 07_cross_type_causal_validation/   # 阶段 8：跨类型因果验证
|   |-- 08_training/                       # 阶段 9：神经元训练
|   |-- 09_evaluation/                     # 阶段 10：指标汇总和评测
|   |-- third_party/                       # 第三方代码适配，不直接混入主逻辑
|       |-- when2tool_adapter/             # When2Tool 原方法适配
|       |   |-- env_schemas/                # When2Tool 官方工具 schema
|       |-- ss_neuron_expansion_adapter/   # Who Transfers Safety? 方法适配
|-- scripts/
    |-- run_01_labeling_demo.sh            # 单模型小样本 demo，检查环境和逻辑
    |-- run_01_labeling_all.sh             # 6 个模型全量生成 0/1 标签
    |-- run_02_build_modified_when2tool.sh # 基于标签生成改造后数据集
    |-- run_03_feature_extraction.sh       # 后续补：一键提特征
    |-- run_04_single_type_neuron_probing.sh
    |-- run_05_single_type_causal_validation.sh
    |-- run_06_shared_neuron_discovery.sh
    |-- run_07_cross_type_causal_validation.sh
    |-- run_08_training.sh
    |-- run_09_evaluation.sh
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
| 模型标签结果 | generated labels | 本项目生成 | 随实验输出交接 | `$DATA_ROOT/labels/<model_alias>/` | 每个模型自己的 `tool_necessary=0/1` 标签 |
| 改造后 When2Tool 数据 | processed dataset | 由原始 When2Tool + 模型标签生成 | 百度网盘链接待补充 | `$DATA_ROOT/datasets/modified_when2tool/<model_alias>/` | 后续探测、因果验证、训练使用 |

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

`requirements.txt` 中 `transformers==4.55.2`、`vllm>=0.8.5` 用于覆盖 Qwen3 / Qwen3-2507 和 Llama Instruct 的官方 Hugging Face / vLLM 推理接口需求。如果当前机器暂时不支持 vLLM，可以先把 `requirements.txt` 里的 vLLM 依赖注释掉；正式跑多模型标签，尤其是 70B 模型时，建议在 Linux GPU 服务器上安装 vLLM。

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

## 阶段 2：标签生成

本阶段对齐 When2Tool 原方法，用原始 When2Tool 数据集生成每个模型自己的 `tool_necessary=0/1` 标签。

注意：这里输入是 `$DATA_ROOT/datasets/raw_when2tool/`。先跑标签，再进入阶段 3 构建改造后数据集。

标签定义：

```text
prompt_mode = hard_no_tool
reasoning_mode = no_reasoning

tool_necessary = 0, 如果模型在不能使用工具时答对
tool_necessary = 1, 如果模型在不能使用工具时答错
```

这和 When2Tool Step 2 / `extract_features.py` 的标签收集逻辑一致：先跑 hard no-tool evaluation，再用 final answer 是否正确得到 no-tool label。

需要覆盖的模型：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```

模型加载使用 Hugging Face Transformers / vLLM 的官方通用接口。Qwen3 普通模型在 `configs/models.yaml` 里显式设置 `enable_thinking: "false"`，用于对齐本阶段的 no reasoning 标签收集；`Qwen3-4B-Instruct-2507` 官方本身是 non-thinking instruct 模型，因此配置为 `auto`，不额外传 thinking 开关；Llama Instruct 模型也不传 Qwen 专用参数。

相关代码：

| 文件 | 作用 |
|---|---|
| `code/01_labeling/build_when2tool_labels.py` | 核心标签生成脚本 |
| `code/01_labeling/README.md` | 本阶段说明 |
| `code/third_party/when2tool_adapter/env_schemas/*.json` | When2Tool 官方工具 schema，用于构造模型看到的 tools |
| `code/common/model_registry.py` | 解析 6 个模型 alias、repo_id 和本地相对路径 |
| `configs/models.yaml` | 6 个模型的路径和 thinking 开关配置 |
| `configs/labeling.yaml` | 标签阶段默认配置记录 |
| `scripts/run_01_labeling_demo.sh` | 单模型小样本 demo |
| `scripts/run_01_labeling_all.sh` | 6 模型全量标签脚本，支持多机分块 |

本阶段禁止自动从 Hugging Face 下载或读缓存：原始数据必须提前放在 `$DATA_ROOT/datasets/raw_when2tool/`，模型权重必须提前放在 `configs/models.yaml` 记录的本地相对路径。缺文件时脚本直接报错。

先检查数据，不加载模型：

```bash
conda activate tool_neurons
python code/01_labeling/build_when2tool_labels.py \
  --data-root ../tool_decision_neurons_data \
  --check-data-only
```

### 单卡

小样本 demo：

```bash
MAX_SAMPLES=5 BACKEND=hf bash scripts/run_01_labeling_demo.sh qwen3-1.7b
```

单卡全量跑一个模型：

```bash
python code/01_labeling/build_when2tool_labels.py \
  --model-alias qwen3-1.7b \
  --data-root ../tool_decision_neurons_data \
  --backend hf \
  --overwrite
```

### 单机八卡

单机 8 卡跑全部 6 个模型：

```bash
BACKEND=vllm TENSOR_PARALLEL_SIZE=8 bash scripts/run_01_labeling_all.sh
```

单机 8 卡只跑指定模型：

```bash
MODEL_ALIASES="qwen3-32b llama3.3-70b" \
BACKEND=vllm TENSOR_PARALLEL_SIZE=8 \
bash scripts/run_01_labeling_all.sh
```

多机分块时，每台机器设置同一个 `NUM_SHARDS` 和不同的 `SHARD_INDEX`：

```bash
MODEL_ALIASES="llama3.3-70b" \
NUM_SHARDS=4 SHARD_INDEX=0 \
BACKEND=vllm TENSOR_PARALLEL_SIZE=8 \
bash scripts/run_01_labeling_all.sh
```

标签输出放在 `$DATA_ROOT/labels/`，每个模型一个文件夹，跑完后可以直接打包对应 `<model_alias>/` 发回。

```text
$DATA_ROOT/
|-- labels/
    |-- <model_alias>/
        |-- manifest.json
        |-- single_hop/
        |   |-- train/
        |   |   |-- labels.jsonl
        |   |   |-- no_tool_outputs.json
        |   |   |-- summary.json
        |   |-- test/
        |       |-- labels.jsonl
        |       |-- no_tool_outputs.json
        |       |-- summary.json
        |-- multi_hop/
            |-- train/
            |   |-- labels.jsonl
            |   |-- no_tool_outputs.json
            |   |-- summary.json
            |-- test/
                |-- labels.jsonl
                |-- no_tool_outputs.json
                |-- summary.json
```

如果使用 `NUM_SHARDS>1`，每个 split 下会多一层 shard 目录，例如：

```text
$DATA_ROOT/labels/<model_alias>/manifest_shard_00000_of_00004.json
$DATA_ROOT/labels/<model_alias>/single_hop/train/shard_00000_of_00004/
|-- labels.jsonl
|-- no_tool_outputs.json
|-- summary.json
```

`labels.jsonl` 每行是一条样本的标签，核心字段：

| 字段 | 含义 |
|---|---|
| `model_alias` | 当前模型 |
| `subset` / `split` | `single_hop` 或 `multi_hop`，`train` 或 `test` |
| `id` / `sample_uid` | When2Tool 原始 id 和全局样本 id |
| `env_name` / `difficulty` | 原始环境和难度 |
| `task_type` | 根据 env 映射得到的 A/B/C，方便后续探测分组 |
| `num_shards` / `shard_index` | 分块信息；不分块时为 `1` / `0` |
| `model_answer_raw` | hard no-tool 模式下模型原始最终回答 |
| `model_answer` | 从 `\boxed{...}` 里抽出的答案 |
| `no_tool_correct` | no-tool 是否答对 |
| `tool_necessary` | 最终 0/1 标签 |

`no_tool_outputs.json` 保存 When2Tool 风格的 hard no-tool 运行轨迹，主要用于检查模型到底是直接答了、被拒绝了工具调用，还是超过轮数没答出来。

## 阶段 3：改造后数据集构建

本阶段在标签产物已经拿到之后运行。它把原始 When2Tool 样本、A/B/C 种类映射、对应模型的 `tool_necessary=0/1` 标签合并成每个模型自己的改造后数据集。

相关代码：

| 文件 | 作用 |
|---|---|
| `code/02_dataset_preparation/build_modified_when2tool.py` | 合并 raw 样本、A/B/C 映射和模型标签 |
| `code/02_dataset_preparation/README.md` | 本阶段说明 |
| `code/common/env_type_mapping.py` | `env_name -> A/B/C` 映射 |
| `scripts/run_02_build_modified_when2tool.sh` | 一键构建改造后数据集 |

输入：

```text
$DATA_ROOT/datasets/raw_when2tool/
$DATA_ROOT/labels/<model_alias>/
```

运行全部已返回标签的模型：

```bash
conda activate tool_neurons
bash scripts/run_02_build_modified_when2tool.sh
```

只运行指定模型：

```bash
MODEL_ALIASES="qwen3-1.7b llama3.1-8b" bash scripts/run_02_build_modified_when2tool.sh
```

输出：

```text
$DATA_ROOT/datasets/modified_when2tool/
|-- manifest.json
|-- env_type_mapping.json
|-- baidu_netdisk_info.md
|-- <model_alias>/
    |-- manifest.json
    |-- label_coverage.csv
    |-- summary.csv
    |-- env_type_mapping.json
    |-- single_hop/
    |   |-- train.jsonl
    |   |-- train.parquet
    |   |-- test.jsonl
    |   |-- test.parquet
    |-- multi_hop/
        |-- train.jsonl
        |-- train.parquet
        |-- test.jsonl
        |-- test.parquet
```

改造后样本核心新增字段：

| 字段 | 含义 |
|---|---|
| `sample_uid` | 全局样本 id，格式为 `subset:split:original_id` |
| `task_type` / `task_type_name` | A/B/C 任务类型 |
| `when2tool_category` | When2Tool 对应任务类别 |
| `model_alias` | 标签所属模型 |
| `tool_necessary` | 模型自己的 0/1 标签 |
| `no_tool_correct` | hard no-tool 是否答对 |
| `model_answer_raw` / `model_answer` | no-tool 原始回答和抽取答案 |

默认要求每个 split 标签完整覆盖原始数据；如果只想用少量样本调试，可加 `--max-samples` 和 `--allow-partial` 直接调用 Python 脚本。

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
01_labeling
02_dataset_preparation
03_feature_extraction
04_single_type_neuron_probing
05_single_type_causal_validation
06_shared_neuron_discovery
07_cross_type_causal_validation
08_training
09_evaluation
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
