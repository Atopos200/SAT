# DSGR 服务器运行指南（换模型版）

本文档给出在服务器上运行 `DSGR` 的可执行步骤，适用于将本地实验迁移到 GPU 服务器并更换为更大模型（如 `Qwen2.5-1.5B`）。

---

## 1. 目标与原则

- 先跑通，再放量：先小样本冒烟，确认无误后再跑全量。
- 先单次，再多种子：先验证单次稳定，再跑 `3-seed` 统计均值/方差。
- 先 `relation` 候选调参，再补 `all` 候选作为论文严格口径。

---

## 2. 环境准备

### 2.1 检查 GPU

```bash
nvidia-smi
```

### 2.2 创建 Python 环境

```bash
conda create -n dsgr python=3.10 -y
conda activate dsgr
```

### 2.3 安装 PyTorch（按 CUDA 版本选择）

示例（CUDA 12.1）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2.4 安装项目依赖

```bash
cd /path/to/sat/SAT
pip install -r requirements.txt
pip install accelerate peft transformers datasets sentencepiece
```

可选：使用新入口运行脚本（与 `run_full_qwen.py` 行为一致）：

```bash
python dsgr/scripts/run_experiment.py
```

### 2.5 配置 HuggingFace（建议）

```bash
export HF_TOKEN=你的token
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
```

---

## 3. 代码配置（`run_full_qwen.py`）

至少调整以下配置：

- `QWEN_MODEL = "Qwen/Qwen2.5-1.5B"`（或你的目标模型）
- `STAGE2_RUN_VARIANT = "DSGR_CoT"`（先只跑创新版）
- `STAGE2_TRAIN_SAMPLES = 5000`（先中等规模，后续可全量）
- `STAGE2_KGC_CANDIDATE_MODE = "relation"`（先调通）

显存不足时，优先降低：

- `STAGE2_MAX_LEN`
- `STAGE2_KGC_SCORE_BATCH`

---

## 4. 先做小样本冒烟

建议先把训练样本临时改为 `500`（或更小），直接前台运行观察：

```bash
python run_full_qwen.py
# or:
python dsgr/scripts/run_experiment.py
```

确认以下信息都正常：

- 日志显示 `使用设备: CUDA`
- `训练样本` 数量符合预期
- 每个 epoch 都有 `Valid Epoch` 指标
- 无 `nan`、无 OOM、无异常退出

---

## 5. 正式后台训练

```bash
nohup python run_full_qwen.py > run_qwen.log 2>&1 &
# or:
nohup python dsgr/scripts/run_experiment.py > run_qwen.log 2>&1 &
tail -f run_qwen.log
```

---

## 6. 断点续训

若任务中断，可使用同一 `RUN_ID` 续训：

```bash
DSGR_RUN_ID=你的run_id DSGR_RESUME=1 python run_full_qwen.py
# or:
DSGR_RUN_ID=你的run_id DSGR_RESUME=1 python dsgr/scripts/run_experiment.py
```

---

## 7. 多种子稳定性实验（推荐论文必做）

```bash
python run_multiseed_qwen.py --seeds 42,43,44 --variant DSGR_CoT
```

输出聚合文件（均值/方差）：

- `results_qwen/runs/<prefix>_<time>_summary.json`

---

## 8. 结果文件说明

单次运行结果目录：

- `results_qwen/runs/<RUN_ID>/config_snapshot.json`：配置快照
- `results_qwen/runs/<RUN_ID>/final_results.json`：最终指标
- `results_qwen/runs/<RUN_ID>/<variant>/predictions_*.json`：逐样本预测

常用核心指标：

- `Hit@1`
- `Hit@3`
- `Hit@10`
- `MRR`
- `skipped`

---

## 9. 论文口径建议

当前默认 `candidate_mode="relation"` 适合快速调参。  
用于论文最终主表，建议补一版：

- `STAGE2_KGC_CANDIDATE_MODE = "all"`

这样更接近标准严格 KGC 全实体排序评估（但更慢）。

---

## 10. 常见问题排查

### 10.1 OOM（显存不足）

优先调整：

1. 降低 `STAGE2_MAX_LEN`
2. 降低 `STAGE2_KGC_SCORE_BATCH`
3. 减少 `STAGE2_TRAIN_SAMPLES` 做分阶段实验

### 10.2 模型下载慢/失败

- 确认 `HF_TOKEN` 设置正确
- 检查网络/代理

### 10.3 指标异常低

- 检查 `data_cot_lp/.../manifest.json` 里 `success/fallback/error_types`
- 抽样查看 `predictions_*.json` 是否在输出解释句而非实体名
- 对比不同 `candidate_mode` 下结果是否一致

---

## 11. 推荐执行顺序（实践版）

1. 小样本冒烟（确认能跑）
2. 单次正式实验（拿到可解释结果）
3. 3-seed 稳定性（均值/方差）
4. `all` 候选严格评估（论文口径）

---

## 12. 一组可复制命令（最小闭环）

```bash
conda activate dsgr
cd /path/to/sat/SAT

# 1) 单次
nohup python run_full_qwen.py > run_qwen.log 2>&1 &
tail -f run_qwen.log

# 2) 多seed
python run_multiseed_qwen.py --seeds 42,43,44 --variant DSGR_CoT
```

