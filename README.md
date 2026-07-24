# Cross-Task Tool-Decision Neurons

基于 When2Tool 和 Who Transfers Safety? 方法，研究 A/B/C 三类任务中影响“是否调用工具”的单类型神经元与跨类型共享神经元。

## To-do List

- [ ] 准备原始 When2Tool 数据集和本地模型权重
- [ ] 对 6 个模型跑 When2Tool 原始 0/1 标签
- [ ] 基于标签构建模型专属改造后数据集
- [ ] 抽取 train/test 特征和激活
- [ ] 在 train split 上探测 A/B/C 单类型工具决策神经元
- [ ] 在 test split 上做单类型因果验证
- [ ] 发现 A/B/C 跨类型共享神经元
- [ ] 在 test split 上做跨类型因果验证
- [ ] 只训练共享神经元
- [ ] 在 test split 上评测训练后模型并汇总

## 章节目录

- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [数据和模型资源](#数据和模型资源)
- [阶段 1：原始数据准备](#阶段-1原始数据准备)
- [阶段 2：模型 0/1 标签生成](#阶段-2模型-01-标签生成)
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

假设当前仓库路径是：

```text
.../tool_decision_neurons_code/
```

大文件、输出、模型权重不提交到 GitHub，放在代码仓库同级目录。

```text
.../
|-- tool_decision_neurons_code/          # GitHub 同步的代码仓库
|   |-- README.md
|   |-- requirements.txt
|   |-- configs/
|   |   |-- models.yaml                  # 模型 alias、repo_id、本地路径
|   |-- code/
|   |   |-- 01_labeling/                 # 阶段 2：跑 When2Tool 0/1 标签
|   |   |-- 02_dataset_preparation/      # 阶段 3：构建改造后数据集
|   |   |-- 03_feature_extraction/       # 阶段 4：抽取表示和激活
|   |   |-- 04_single_type_neuron_probing/# 阶段 5：探测 A/B/C 单类型神经元
|   |   |-- 05_single_type_causal_validation/ # 阶段 6：单类型因果验证
|   |   |-- 06_shared_neuron_discovery/   # 阶段 7：找共享神经元
|   |   |-- 07_cross_type_causal_validation/ # 阶段 8：跨类型因果验证
|   |   |-- 08_training/                 # 阶段 9：只训练共享神经元
|   |   |-- 09_evaluation/               # 阶段 10：训练后评测和汇总
|   |   |-- 10_multigpu/                 # 阶段 4-10：单机多卡 Python 调度入口
|   |   |-- common/                      # 公共 IO、映射、多卡调度、画图工具
|   |   |-- third_party/when2tool_adapter/# When2Tool env/schema 适配
|   |-- scripts/                        # 阶段 1-3 已有脚本，后续可自行调整
|
|-- tool_decision_neurons_data/          # 不提交 GitHub
|   |-- datasets/
|   |   |-- raw_when2tool/               # 原始 When2Tool 数据集
|   |   |-- modified_when2tool/          # 阶段 3 输出：每个模型一份改造后数据集
|   |-- labels/                         # 阶段 2 输出：每个模型的 0/1 标签
|   |-- features/                       # 阶段 4 输出：表示和激活
|   |-- neurons/                        # 阶段 5/7 输出：单类型和共享神经元
|   |-- causal_validation/              # 阶段 6/8 输出：因果验证结果
|   |-- training/                       # 阶段 9 输出：共享神经元 delta checkpoint
|   |-- outputs/                        # 阶段 10 输出：最终评测汇总
|   |-- visualizations/                 # 阶段 6/7/8 图
|
|-- Qwen/                               # 不提交 GitHub
|   |-- qwen3-1.7b/
|   |-- qwen3-4b-instruct/
|   |-- qwen3-14b/
|   |-- qwen3-32b/
|
|-- meta-llama/                         # 不提交 GitHub
    |-- llama3.1-8b/
    |-- llama3.3-70b/
```

## 环境配置

```bash
conda create -n tool_neurons python=3.10 -y
conda activate tool_neurons

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install modelscope
```

## 数据和模型资源

先进入仓库根目录：

```bash
cd .../tool_decision_neurons_code
mkdir -p ../tool_decision_neurons_data/datasets/raw_when2tool
```

下载原始 When2Tool 数据集：

```bash
huggingface-cli download cesun/When2Tool \
  --repo-type dataset \
  --local-dir ../tool_decision_neurons_data/datasets/raw_when2tool \
  --local-dir-use-symlinks False
```

下载模型权重。代码默认禁止自动从 Hugging Face 缓存下载，所有模型都要提前放到 `configs/models.yaml` 对应的本地路径。

```bash
mkdir -p ../Qwen ../meta-llama

modelscope download --model Qwen/Qwen3-1.7B --local_dir ../Qwen/qwen3-1.7b
modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir ../Qwen/qwen3-4b-instruct
modelscope download --model Qwen/Qwen3-14B --local_dir ../Qwen/qwen3-14b
modelscope download --model Qwen/Qwen3-32B --local_dir ../Qwen/qwen3-32b

modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --local_dir ../meta-llama/llama3.1-8b
modelscope download --model LLM-Research/Llama-3.3-70B-Instruct --local_dir ../meta-llama/llama3.3-70b
```

## 阶段 1：原始数据准备

输入：

```text
../tool_decision_neurons_data/datasets/raw_when2tool/
```

该目录只放 When2Tool 原始数据，后续阶段 2 跑标签时直接读取这里，不读取改造后数据集。

## 阶段 2：模型 0/1 标签生成

阶段 2 使用 When2Tool 原始数据集跑模型的 hard-no-tool 结果，得到：

```text
tool_necessary = 0  表示模型无工具条件下答对
tool_necessary = 1  表示模型无工具条件下答错，因此该样本对该模型需要工具
```

示例命令：

```bash
python code/01_labeling/build_when2tool_labels.py \
  --model-alias qwen3-4b-instruct \
  --model-path ../Qwen/qwen3-4b-instruct \
  --data-root ../tool_decision_neurons_data \
  --subsets single_hop multi_hop \
  --splits train test \
  --torch-dtype bfloat16 \
  --device-map auto
```

输出：

```text
../tool_decision_neurons_data/labels/<model_alias>/
|-- single_hop/
|   |-- train/labels.jsonl
|   |-- test/labels.jsonl
|-- multi_hop/
    |-- train/labels.jsonl
    |-- test/labels.jsonl
```

## 阶段 3：改造后数据集构建

阶段 3 读取原始 When2Tool 数据和阶段 2 生成的模型标签，构建模型专属改造后数据集。每条样本保留原始字段，并加入 `task_type=A/B/C` 和 `tool_necessary=0/1`。

```bash
python code/02_dataset_preparation/build_modified_when2tool.py \
  --data-root ../tool_decision_neurons_data \
  --model-aliases qwen3-4b-instruct \
  --subsets single_hop multi_hop \
  --splits train test \
  --overwrite
```

输出：

```text
../tool_decision_neurons_data/datasets/modified_when2tool/<model_alias>/
|-- manifest.json
|-- single_hop/
|   |-- train.jsonl
|   |-- test.jsonl
|-- multi_hop/
    |-- train.jsonl
    |-- test.jsonl
```

## 阶段 4：特征和激活提取

阶段 4 读取阶段 3 的改造后数据集，构造 When2Tool current/no-reasoning prompt，保存 Who Transfers Safety? 探测需要的最后 token 表示 `z_m(x)`，并保存 Q/K/V/O 投影激活用于维度检查和审计。

单跳、多跳、train/test 都提取；阶段 5 只会使用 train split。

```bash
python code/10_multigpu/run_stage4_extract_features_8gpu.py \
  --model-alias qwen3-4b-instruct \
  --model-path ../Qwen/qwen3-4b-instruct \
  --modified-dir ../tool_decision_neurons_data/datasets/modified_when2tool/qwen3-4b-instruct \
  --output-dir ../tool_decision_neurons_data/features \
  --subsets single_hop multi_hop \
  --splits train test \
  --num-gpus 8 \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --torch-dtype bfloat16 \
  --device-map auto \
  --enable-thinking auto
```

多卡方式：按样本行 round-robin 分成 8 个 shard，每张卡一个 worker，最后合并回正式 split 目录。若目标 split 已有完整 `activations.pt/meta.jsonl/summary.json`，默认提前跳过；需要重跑时加 `--overwrite`。

输出：

```text
../tool_decision_neurons_data/features/<model_alias>/
|-- manifest.json
|-- single_hop/
|   |-- train/
|   |   |-- activations.pt
|   |   |-- meta.jsonl
|   |   |-- summary.json
|   |-- test/
|       |-- activations.pt
|       |-- meta.jsonl
|       |-- summary.json
|-- multi_hop/
    |-- train/
    |-- test/
```

## 阶段 5：单类型神经元探测

阶段 5 只在 train split 上探测神经元，并且 single_hop / multi_hop 分开探测。

神经元定义直接对齐 Who Transfers Safety?：`N=(layer, matrix, index)`，`matrix` 属于 `Q/K/V/O`。Q/K/V 神经元对应 `W_Q/W_K/W_V` 的行，O 神经元对应 `W_O` 的列。重要性为：

```text
Delta_m(x,N)=||z_m(x)-z_{m,without N}(x)||_2
I_m(N,D)=mean_{x in D} Delta_m(x,N)
```

对每个任务类型 `c`，先得到 `TopP(I(N,D^1_{m,c}))` 和 `TopP(I(N,D^0_{m,c}))`，再取差集：

```text
TDN_{m,c,l}=TopP(I(N,D^1_{m,c,l})) - TopP(I(N,D^0_{m,c,l}))
```

```bash
python code/10_multigpu/run_stage5_probe_single_type_8gpu.py \
  --model-alias qwen3-4b-instruct \
  --model-path ../Qwen/qwen3-4b-instruct \
  --feature-dir ../tool_decision_neurons_data/features/qwen3-4b-instruct \
  --output-dir ../tool_decision_neurons_data/neurons \
  --subsets single_hop multi_hop \
  --probe-splits train \
  --num-gpus 8 \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --top-p 0.03 \
  --deactivation-batch-size 1 \
  --torch-dtype bfloat16 \
  --device-map auto
```

多卡方式：不是切数据，而是切候选神经元。对每个 subset，完整候选神经元集合按 `candidate_idx % 8` 分成 8 份，每张卡加载一个模型，只计算自己的候选神经元 deactivation score。8 个 `score_shard.pt` 完成后，runner 检查候选神经元无重复、无缺失，再在完整每层候选集合上做 TopP 和 `S1 - S0`，输出正式 A/B/C 神经元文件。只有带有当前候选分片合并签名的正式 subset 才会被判定为完成；旧版本完整产物、半成品、分片数不一致的产物都会被视为 stale，并在重跑前自动清理。中断后重跑时，已完成且签名正确的 subset 会提前跳过。

输出：

```text
../tool_decision_neurons_data/neurons/<model_alias>/single_type_by_subset/
|-- manifest.json
|-- single_hop/
|   |-- manifest.json
|   |-- A/
|   |   |-- TDN_neurons.jsonl
|   |   |-- S1_top_neurons.jsonl
|   |   |-- S0_top_neurons.jsonl
|   |   |-- scores_and_masks.pt
|   |   |-- summary.json
|   |-- B/
|   |-- C/
|-- multi_hop/
    |-- A/
    |-- B/
    |-- C/
```

临时中间产物：

```text
../tool_decision_neurons_data/neurons/<model_alias>/single_type_by_subset/_stage5_workers/
|-- single_hop/shard_00000/.../score_shard.pt
|-- ...
|-- multi_hop/shard_00007/.../score_shard.pt
```

默认合并成功后会清理 `_stage5_workers/`；需要保留中间分片时加 `--keep-workdir`。

## 阶段 6：单类型因果验证

阶段 6 使用阶段 5 在 train split 探测出的 A/B/C 单类型神经元，在 test split 上做因果验证。single_hop / multi_hop 分开评测。

每个任务类型分别跑 `Base`、`M-Random`、`M-TDN_c`。指标对齐 When2Tool：`Acc`、`TC`、`AvgTC`、`TCR`，并保留工具决策指标 `ToolAcc`、`ToolNecessaryAcc`、`NoToolAcc`、`OverCall`。

```bash
python code/10_multigpu/run_stage6_single_type_causal_validation_8gpu.py \
  --model-alias qwen3-4b-instruct \
  --model-path ../Qwen/qwen3-4b-instruct \
  --modified-dir ../tool_decision_neurons_data/datasets/modified_when2tool/qwen3-4b-instruct \
  --neuron-dir ../tool_decision_neurons_data/neurons/qwen3-4b-instruct/single_type_by_subset \
  --output-dir ../tool_decision_neurons_data/causal_validation \
  --subsets single_hop multi_hop \
  --task-types A B C \
  --split test \
  --num-gpus 8 \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --max-rounds 10 \
  --max-new-tokens 2048 \
  --record-mode lite
```

多卡方式：按 `subset × task_type × intervention` 拆成 18 个单卡 worker，最多同时跑 8 个。每个 worker 只写一个 intervention 目录，全部完成后自动 refresh 生成汇总表和图。每个 intervention 会额外保存 `runner_meta.json`，记录当前 Stage 5 subset manifest 的 hash；中断后重跑时，只有文件完整且 hash 一致的 unit 会跳过。旧版本产物、半成品、依赖神经元已变化的产物都会在重跑前自动清理。

输出：

```text
../tool_decision_neurons_data/causal_validation/<model_alias>/single_type_by_subset/
|-- manifest.json
|-- summary_table.csv
|-- summary.md
|-- all_per_task.jsonl
|-- single_hop/
|   |-- summary_table.csv
|   |-- summary.md
|   |-- figures/
|   |-- A/
|   |   |-- Base/per_task.jsonl
|   |   |-- M-Random/per_task.jsonl
|   |   |-- M-TDN_A/per_task.jsonl
|   |-- B/
|   |-- C/
|-- multi_hop/
```

## 阶段 7：共享神经元发现

阶段 7 读取阶段 5 的 A/B/C 单类型 TDN，按照 Who Transfers Safety? 的交集思想找跨任务类型共享神经元：

```text
CTD_{m,l}=TDN_{m,A,l} ∩ TDN_{m,B,l} ∩ TDN_{m,C,l}
CTD_m=union_l CTD_{m,l}
```

交集按完整神经元身份 `(layer, matrix, index)` 精确匹配，不做占位或替代。

```bash
python code/06_shared_neuron_discovery/discover_shared_neurons.py \
  --model-alias qwen3-4b-instruct \
  --data-root ../tool_decision_neurons_data \
  --single-type-dir ../tool_decision_neurons_data/neurons/qwen3-4b-instruct/single_type_by_subset \
  --causal-dir ../tool_decision_neurons_data/causal_validation/qwen3-4b-instruct/single_type_by_subset \
  --output-dir ../tool_decision_neurons_data/neurons \
  --make-figures \
  --skip-existing
```

该阶段是纯数据处理和画图，不占 GPU。`--skip-existing` 会在最终 manifest 已存在时提前跳过。

输出：

```text
../tool_decision_neurons_data/neurons/<model_alias>/shared_by_subset/
|-- manifest.json
|-- shared_summary.csv
|-- single_hop/
|   |-- CTD_neurons.jsonl
|   |-- pairwise_AB_neurons.jsonl
|   |-- pairwise_AC_neurons.jsonl
|   |-- pairwise_BC_neurons.jsonl
|   |-- private_A_neurons.jsonl
|   |-- private_B_neurons.jsonl
|   |-- private_C_neurons.jsonl
|   |-- layer_counts.csv
|   |-- matrix_counts.csv
|   |-- share_rates.csv
|   |-- manifest.json
|-- multi_hop/
```

图输出：

```text
../tool_decision_neurons_data/visualizations/<model_alias>/shared_by_subset/
|-- fig4_shared_abundance_vs_capability.png
|-- fig_shared_neuron_heatmap.png
|-- fig4_share_rate_vs_type_capability_single_hop.png
|-- fig4_share_rate_vs_type_capability_multi_hop.png
|-- summary.md
|-- manifest.json
```

## 阶段 8：跨类型因果验证

阶段 8 使用阶段 7 的共享神经元，在 test split 上做跨类型因果验证。single_hop / multi_hop 分开评测。

每个任务类型跑 `Base`、`M-Random`、`M-CTD`、`M-Private_c`。`M-Private_c` 是该类型单类型 TDN 去掉共享 CTD 后的私有神经元，用来验证真正跨类型负责的是共享神经元。

```bash
python code/10_multigpu/run_stage8_cross_type_causal_validation_8gpu.py \
  --model-alias qwen3-4b-instruct \
  --model-path ../Qwen/qwen3-4b-instruct \
  --modified-dir ../tool_decision_neurons_data/datasets/modified_when2tool/qwen3-4b-instruct \
  --shared-dir ../tool_decision_neurons_data/neurons/qwen3-4b-instruct/shared_by_subset \
  --single-type-dir ../tool_decision_neurons_data/neurons/qwen3-4b-instruct/single_type_by_subset \
  --output-dir ../tool_decision_neurons_data/causal_validation \
  --subsets single_hop multi_hop \
  --task-types A B C \
  --split test \
  --num-gpus 8 \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --max-rounds 10 \
  --max-new-tokens 2048 \
  --record-mode lite
```

多卡方式：按 `subset × task_type × intervention` 拆成 24 个单卡 worker，最多同时跑 8 个。全部完成后自动 refresh 汇总表格和图。每个 intervention 会额外保存 `runner_meta.json`，记录当前 Stage 7 shared manifest 和 Stage 5 single-type manifest 的 hash；中断后重跑时，只有文件完整且依赖 hash 一致的 unit 会跳过。旧版本产物、半成品、依赖已变化的产物都会在重跑前自动清理。

输出：

```text
../tool_decision_neurons_data/causal_validation/<model_alias>/cross_type_by_subset/
|-- manifest.json
|-- summary_table.csv
|-- cross_type_effects.csv
|-- summary.md
|-- figures/
|-- single_hop/
|   |-- summary_table.csv
|   |-- cross_type_effects.csv
|   |-- summary.md
|   |-- figures/
|   |-- A/
|   |   |-- Base/per_task.jsonl
|   |   |-- M-Random/per_task.jsonl
|   |   |-- M-CTD/per_task.jsonl
|   |   |-- M-Private_A/per_task.jsonl
|   |-- B/
|   |-- C/
|-- multi_hop/
```

## 阶段 9：神经元训练

阶段 9 只训练阶段 7 找到的 CTD 共享神经元参数。训练数据来自阶段 3 的 train split，single_hop / multi_hop 分开训练并分别保存 delta checkpoint。

训练方式对齐 Who Transfers Safety? 的神经元训练思想：冻结普通参数，只给 CTD 对应的 Q/K/V 行和 O 列保留梯度；loss 是 assistant token 的自回归交叉熵，system/user/tool_response token 不计入 loss。

```bash
python code/10_multigpu/run_stage9_train_shared_neurons_8gpu.py \
  --model-alias qwen3-4b-instruct \
  --model-path ../Qwen/qwen3-4b-instruct \
  --modified-dir ../tool_decision_neurons_data/datasets/modified_when2tool/qwen3-4b-instruct \
  --shared-dir ../tool_decision_neurons_data/neurons/qwen3-4b-instruct/shared_by_subset \
  --output-dir ../tool_decision_neurons_data/training \
  --subsets single_hop multi_hop \
  --task-types A B C \
  --split train \
  --num-gpus 8 \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.03 \
  --max-length 2048 \
  --torch-dtype bfloat16 \
  --device-map auto
```

说明：single_hop / multi_hop 是两个独立训练目标。该阶段不能像 Stage 5 那样把 CTD 神经元切成 8 份分别训练再合并，否则训练目标不等价。已有完整 subset checkpoint 时默认提前跳过。

输出：

```text
../tool_decision_neurons_data/training/<model_alias>/neuron_training_by_subset/
|-- manifest.json
|-- single_hop/
|   |-- ctd_neuron_delta.pt
|   |-- training_log.csv
|   |-- training_examples.jsonl
|   |-- skipped_examples.jsonl
|   |-- trainable_mask_summary.json
|   |-- manifest.json
|   |-- summary.md
|-- multi_hop/
```

## 阶段 10：评测和汇总

阶段 10 加载阶段 9 的 CTD delta checkpoint，在 test split 上重新评测，并和阶段 8 的 `Base` 结果汇总对比。指标继续对齐 When2Tool：`Acc`、`TC`、`AvgTC`、`TCR`，并保留工具决策指标 `ToolAcc`、`ToolNecessaryAcc`、`NoToolAcc`。`AUROC` 在本阶段没有额外 probe 分数时记为 `NA`。

```bash
python code/10_multigpu/run_stage10_evaluate_training_summary_8gpu.py \
  --model-alias qwen3-4b-instruct \
  --model-path ../Qwen/qwen3-4b-instruct \
  --modified-dir ../tool_decision_neurons_data/datasets/modified_when2tool/qwen3-4b-instruct \
  --training-dir ../tool_decision_neurons_data/training/qwen3-4b-instruct/neuron_training_by_subset \
  --default-eval-dir ../tool_decision_neurons_data/causal_validation/qwen3-4b-instruct/cross_type_by_subset \
  --output-dir ../tool_decision_neurons_data/outputs \
  --subsets single_hop multi_hop \
  --task-types A B C \
  --split test \
  --num-gpus 8 \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --max-rounds 10 \
  --max-new-tokens 2048 \
  --record-mode lite
```

输出：

```text
../tool_decision_neurons_data/outputs/<model_alias>/training_summary_by_subset/
|-- manifest.json
|-- summary.md
|-- training_comparison_by_type.csv
|-- training_comparison_summary.csv
|-- training_delta_summary.csv
|-- applied_delta_summary.json
|-- all_per_task.jsonl
|-- single_hop/
|   |-- A/
|   |   |-- CTD-training/
|   |       |-- per_task.jsonl
|   |       |-- outputs.json
|   |       |-- summary.json
|   |-- B/
|   |-- C/
|-- multi_hop/
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

所有阶段输出都必须包含 `<model_alias>`，不同模型不能相互覆盖。

## 参考链接

- Who Transfers Safety?: https://arxiv.org/abs/2602.01283
- LLM Agents Already Know When to Call Tools: https://arxiv.org/abs/2605.09252
- When2Tool dataset: https://huggingface.co/datasets/cesun/When2Tool
- When2Tool code: https://github.com/Trustworthy-ML-Lab/when2tool
- Qwen3 models: https://modelscope.cn/organization/Qwen
- Llama ModelScope mirrors: https://modelscope.cn/organization/LLM-Research
