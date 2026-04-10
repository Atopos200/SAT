"""
DSGR 全流程：用 Qwen2.5-0.5B 跑出 Hit@1/MRR（自动检测 CUDA/MPS/CPU）
包含: Stage1(Aligner) -> CoT数据生成 -> Stage2(Qwen微调) -> 评估

本地 Mac: conda activate sat_cpu && cd SAT && python run_full_qwen.py
服务器 GPU: 传代码+数据到服务器 -> 创建环境装依赖 -> nohup python run_full_qwen.py &
"""
import os
import sys
import json
import copy
import random
import logging
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from tqdm import tqdm
from dsgr.data.dataset import KGCDataset
from dsgr.train.evaluate import (
    score_candidate_entities as eval_score_candidate_entities,
    build_filtered_candidate_ids as eval_build_filtered_candidate_ids,
    evaluate_ranking_dataset as eval_evaluate_ranking_dataset,
)
from dsgr.train.checkpoint import (
    get_trainable_state_dict as ckpt_get_trainable_state_dict,
    load_trainable_state_dict as ckpt_load_trainable_state_dict,
    save_runtime_snapshot as ckpt_save_runtime_snapshot,
)
from dsgr.train.trainer import train_variant_with_eval

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

DATA_DIR = os.path.join("aligner", "data", "FB15k-237N")
CKPT_DIR = os.path.join("checkpoints_cpu", "FB15k-237N")
PRED_DATA = os.path.join("predictor", "data_llm_lp", "FB15k-237N")
QWEN_MODEL = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = os.path.join("results_qwen")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

GLOBAL_SEED = int(os.getenv("DSGR_SEED", "42"))
RUN_ID = os.getenv("DSGR_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
RESUME_TRAIN = os.getenv("DSGR_RESUME", "0") == "1"

# --- 配置 ---
STAGE1_EPOCHS = 5
STAGE1_BATCH = 32
STAGE1_LR = 1e-4
STAGE1_DIM = 64
STAGE1_T_LAYERS = 4
STAGE1_T_WIDTH = 128
STAGE1_GT_LAYERS = 2
STAGE1_MAX_QUERIES = 3000

STAGE2_EPOCHS = 3
STAGE2_BATCH = 1
STAGE2_LR = 5e-5
STAGE2_MAX_LEN = 384
STAGE2_TRAIN_SAMPLES = 5000
STAGE2_EVAL_SAMPLES = 500
STAGE2_VALID_SAMPLES = 300
STAGE2_GRAD_ACCUM = 4
# 标准 KGC 评估协议（filtered ranking）
STAGE2_KGC_CANDIDATE_MODE = "relation"   # "all" | "relation" | "subgraph"
STAGE2_KGC_MAX_CANDIDATES = 0            # 0 表示不截断；>0 会限制候选数（用于低算力调试）
STAGE2_KGC_SCORE_BATCH = 32
STAGE2_KGC_GRAPH_SUPPORT_WEIGHT = 0.15
# 只跑某个 variant：设为 "original_SAT" 或 "DSGR_CoT"；设为 None 则两个都跑
STAGE2_RUN_VARIANT = "DSGR_CoT"


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(GLOBAL_SEED)

# ==================== 数据加载工具 ====================
def load_id_map(path):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2: d[parts[0]] = int(parts[1])
    return d

def load_id2text(path):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2: d[int(parts[0])] = parts[1]
    return d

def load_triples(data_path, splits, ent2id, rel2id):
    triples = []
    for split in splits:
        path = os.path.join(data_path, f"{split}.txt")
        if not os.path.exists(path): continue
        with open(path) as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p)==3 and p[0] in ent2id and p[1] in rel2id and p[2] in ent2id:
                    triples.append((ent2id[p[0]], rel2id[p[1]], ent2id[p[2]]))
    return triples


def build_entity_names(id2text):
    names = {}
    for eid, text in id2text.items():
        first_sent = text.split('.')[0].strip()
        if len(first_sent) > 100:
            first_sent = first_sent[:97] + "..."
        names[eid] = first_sent if first_sent else f"Entity_{eid}"
    return names


def get_hrt_from_item(item):
    graph = item.get("graph", {})
    node_idx = graph.get("node_idx", None)
    if isinstance(node_idx, list) and len(node_idx) >= 3:
        return int(node_idx[0]), int(node_idx[1]), int(node_idx[2])
    return None


def score_candidate_entities(model_v, tokenizer, device, prompt, candidate_texts, max_len, batch_size=32):
    return eval_score_candidate_entities(
        model_v=model_v,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        candidate_texts=candidate_texts,
        max_len=max_len,
        batch_size=batch_size,
    )


def build_filtered_candidate_ids(item, h_id, r_id, t_id, true_tails_by_hr, tails_by_rel, all_entity_ids,
                                 mode="relation", max_candidates=0):
    return eval_build_filtered_candidate_ids(
        item=item,
        h_id=h_id,
        r_id=r_id,
        t_id=t_id,
        true_tails_by_hr=true_tails_by_hr,
        tails_by_rel=tails_by_rel,
        all_entity_ids=all_entity_ids,
        mode=mode,
        max_candidates=max_candidates,
    )


def collect_runtime_config():
    keys = [k for k in globals().keys() if k.startswith("STAGE1_") or k.startswith("STAGE2_")]
    cfg = {k: globals()[k] for k in sorted(keys)}
    cfg.update({
        "GLOBAL_SEED": GLOBAL_SEED,
        "RUN_ID": RUN_ID,
        "RUN_DIR": RUN_DIR,
        "RESUME_TRAIN": RESUME_TRAIN,
        "QWEN_MODEL": QWEN_MODEL,
        "DATA_DIR": DATA_DIR,
        "CKPT_DIR": CKPT_DIR,
        "PRED_DATA": PRED_DATA,
    })
    return cfg


def save_runtime_snapshot():
    return ckpt_save_runtime_snapshot(RUN_DIR, collect_runtime_config())


def get_trainable_state_dict(model_v):
    return ckpt_get_trainable_state_dict(model_v)


def load_trainable_state_dict(model_v, state_dict_path):
    return ckpt_load_trainable_state_dict(model_v, state_dict_path)


# ==================== Stage 1: Aligner (如果已有 checkpoint 则跳过) ====================
def run_stage1():
    logging.info("=" * 60)
    logging.info("  STAGE 1: Aligner Training")
    logging.info("=" * 60)

    best_model_path = os.path.join(CKPT_DIR, "aligner_best.pkl")
    emb_path = os.path.join(CKPT_DIR, "entity_embedding.pt")
    graph_path = os.path.join(CKPT_DIR, "graph_data_all.pt")

    if os.path.exists(best_model_path) and os.path.exists(emb_path) and os.path.exists(graph_path):
        logging.info("  Stage 1 checkpoint 已存在，跳过训练")
        logging.info(f"  使用: {best_model_path}")
        return

    logging.info("  运行 test_aligner_cpu.py ...")
    os.system(f"{sys.executable} test_aligner_cpu.py")
    assert os.path.exists(best_model_path), "Stage 1 训练失败"
    logging.info("  Stage 1 完成")


# ==================== Stage 1.5: CoT 数据生成 ====================
def run_cot_generation():
    logging.info("=" * 60)
    logging.info("  STAGE 1.5: CoT Data Generation")
    logging.info("=" * 60)

    cot_path = os.path.join("data_cot_lp", "FB15k-237N", "sample_cot.json")
    if os.path.exists(cot_path):
        logging.info(f"  CoT 数据已存在: {cot_path}")
        return

    logging.info("  运行 test_innovation_cpu.py ...")
    os.system(f"{sys.executable} test_innovation_cpu.py")
    assert os.path.exists(cot_path), "CoT 数据生成失败"
    logging.info("  CoT 数据生成完成")


def extract_answer(text, tokenizer):
    """从生成文本中提取答案"""
    if "Answer:" in text:
        ans = text.split("Answer:")[-1].strip()
    else:
        ans = text.strip()
    ans = ans.split("\n")[0].strip()
    ans = ans.split("</s>")[0].strip()
    ans = ans.split("<|endoftext|>")[0].strip()
    return ans


def normalize(s):
    """标准化文本用于匹配"""
    import re, string
    s = s.lower().strip()
    exclude = set(string.punctuation)
    s = "".join(c for c in s if c not in exclude)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def is_soft_hit(pred_norm, gt_norm):
    """宽松匹配，但避免空预测或极短片段误判为命中。"""
    if not pred_norm or not gt_norm:
        return False
    if pred_norm == gt_norm:
        return True
    return gt_norm in pred_norm or pred_norm in gt_norm


def evaluate_ranking_dataset(model_v, tokenizer, device, eval_data, true_tails_by_hr, tails_by_rel,
                             all_entity_ids, entity_names):
    return eval_evaluate_ranking_dataset(
        model_v=model_v,
        tokenizer=tokenizer,
        device=device,
        eval_data=eval_data,
        true_tails_by_hr=true_tails_by_hr,
        tails_by_rel=tails_by_rel,
        all_entity_ids=all_entity_ids,
        entity_names=entity_names,
        get_hrt_from_item=get_hrt_from_item,
        max_len=STAGE2_MAX_LEN,
        candidate_mode=STAGE2_KGC_CANDIDATE_MODE,
        max_candidates=STAGE2_KGC_MAX_CANDIDATES,
        score_batch=STAGE2_KGC_SCORE_BATCH,
    )


def run_stage2():
    logging.info("=" * 60)
    logging.info("  STAGE 2: Qwen2.5-0.5B Fine-tuning + Evaluation")
    logging.info("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # --- 加载 tokenizer 和模型 ---
    logging.info(f"  下载并加载 {QWEN_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, trust_remote_code=True, torch_dtype=torch.float32
    )
    model_size = sum(p.numel() for p in model.parameters())
    logging.info(f"  模型参数: {model_size:,} ({model_size*4/1024/1024:.0f}MB FP32)")

    # --- LoRA ---
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"  LoRA 可训练参数: {trainable:,} ({trainable*100/model_size:.2f}%)")

    # --- 设备: 有 GPU 用 CUDA，Mac 用 MPS，否则 CPU ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"  使用设备: CUDA ({torch.cuda.get_device_name(0)})")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("  使用设备: MPS (Apple)")
    else:
        device = torch.device("cpu")
        logging.info("  使用设备: CPU")
    model = model.to(device)

    # --- 准备两份数据: 原版 + CoT ---
    orig_train_path = os.path.join(PRED_DATA, "train.json")
    orig_test_path = os.path.join(PRED_DATA, "test.json")
    cot_train_path = os.path.join("data_cot_lp", "FB15k-237N", "train.json")

    # --- 标准 KGC filtered ranking 所需数据 ---
    ent2id = load_id_map(os.path.join(DATA_DIR, "mid2id.txt"))
    rel2id = load_id_map(os.path.join(DATA_DIR, "rel2id.txt"))
    id2text = load_id2text(os.path.join(DATA_DIR, "id2text.txt"))
    entity_names = build_entity_names(id2text)
    triples_all = load_triples(DATA_DIR, ["train", "valid", "test"], ent2id, rel2id)
    true_tails_by_hr = defaultdict(set)
    tails_by_rel = defaultdict(set)
    for h, r, t in triples_all:
        true_tails_by_hr[(h, r)].add(t)
        tails_by_rel[r].add(t)
    all_entity_ids = sorted(entity_names.keys())

    results = {}
    variants = [("original_SAT", orig_train_path), ("DSGR_CoT", cot_train_path)]
    if STAGE2_RUN_VARIANT is not None:
        variants = [(n, p) for n, p in variants if n == STAGE2_RUN_VARIANT]
        logging.info(f"  仅运行 variant: {STAGE2_RUN_VARIANT}")
    if not variants:
        raise ValueError(f"未匹配到可运行的 variant: {STAGE2_RUN_VARIANT}")

    with open(orig_test_path) as f:
        test_data = json.load(f)[:STAGE2_EVAL_SAMPLES]
    valid_path = os.path.join(PRED_DATA, "valid.json")
    with open(valid_path) as f:
        valid_data = json.load(f)[:STAGE2_VALID_SAMPLES]

    for variant_name, train_path in variants:
        logging.info(f"\n{'─'*50}")
        logging.info(f"  Training variant: {variant_name}")
        logging.info(f"  Data: {train_path}")
        logging.info(f"{'─'*50}")

        if variant_name == "DSGR_CoT":
            model_fresh = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL, trust_remote_code=True, torch_dtype=torch.float32
            )
            model_v = get_peft_model(model_fresh, lora_config).to(device)
        else:
            model_v = model

        train_ds = KGCDataset(train_path, tokenizer, STAGE2_MAX_LEN, STAGE2_TRAIN_SAMPLES)
        train_loader = DataLoader(train_ds, batch_size=STAGE2_BATCH, shuffle=True)
        logging.info(f"  训练样本: {len(train_ds)}")

        variant_dir = os.path.join(RUN_DIR, variant_name)
        os.makedirs(variant_dir, exist_ok=True)
        eval_kwargs = {
            "true_tails_by_hr": true_tails_by_hr,
            "tails_by_rel": tails_by_rel,
            "all_entity_ids": all_entity_ids,
            "entity_names": entity_names,
            "get_hrt_from_item": get_hrt_from_item,
            "max_len": STAGE2_MAX_LEN,
            "candidate_mode": STAGE2_KGC_CANDIDATE_MODE,
            "max_candidates": STAGE2_KGC_MAX_CANDIDATES,
            "score_batch": STAGE2_KGC_SCORE_BATCH,
            "graph_support_weight": STAGE2_KGC_GRAPH_SUPPORT_WEIGHT,
        }
        train_info = train_variant_with_eval(
            model_v=model_v,
            tokenizer=tokenizer,
            device=device,
            train_loader=train_loader,
            valid_data=valid_data,
            eval_kwargs=eval_kwargs,
            variant_dir=variant_dir,
            variant_name=variant_name,
            epochs=STAGE2_EPOCHS,
            grad_accum=STAGE2_GRAD_ACCUM,
            lr=STAGE2_LR,
            resume=RESUME_TRAIN,
        )
        ckpt_best_lora = train_info["ckpt_best_lora"]
        best_epoch = train_info["best_epoch"]
        best_valid_mrr = train_info["best_valid_mrr"]
        logging.info(
            f"  训练完成: last_loss={train_info['avg_loss_last_epoch']:.4f}, "
            f"best_epoch={best_epoch}, best_valid_mrr={best_valid_mrr:.2f}"
        )

        # --- 评估 ---
        logging.info(f"  评估中...")
        model_v.eval()
        if os.path.exists(ckpt_best_lora):
            load_trainable_state_dict(model_v, ckpt_best_lora)
            logging.info(f"  使用 best checkpoint (epoch={best_epoch}, valid_mrr={best_valid_mrr:.2f}) 做测试评估")
        test_metrics, predictions = evaluate_ranking_dataset(
            model_v=model_v,
            tokenizer=tokenizer,
            device=device,
            eval_data=test_data,
            true_tails_by_hr=true_tails_by_hr,
            tails_by_rel=tails_by_rel,
            all_entity_ids=all_entity_ids,
            entity_names=entity_names,
        )
        test_metrics["best_epoch"] = best_epoch
        test_metrics["best_valid_mrr"] = round(best_valid_mrr, 2) if best_valid_mrr >= 0 else None
        results[variant_name] = test_metrics

        logging.info(f"\n  [{variant_name}] Results:")
        logging.info(f"    Protocol: filtered ranking, candidate_mode={STAGE2_KGC_CANDIDATE_MODE}")
        logging.info(f"    Hit@1: {results[variant_name]['Hit@1']:.2f}%")
        logging.info(f"    Hit@3: {results[variant_name]['Hit@3']:.2f}%")
        logging.info(f"    Hit@10:{results[variant_name]['Hit@10']:.2f}%")
        logging.info(f"    MRR:   {results[variant_name]['MRR']:.2f}%")
        logging.info(f"    Total: {results[variant_name]['total']}, Skipped: {results[variant_name]['skipped']}")

        pred_path = os.path.join(variant_dir, f"predictions_{variant_name}.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        logging.info(f"    Predictions saved: {pred_path}")

        del model_v
        import gc; gc.collect()

    return results


# ==================== 最终报告 ====================
def final_report(results):
    logging.info("\n" + "=" * 60)
    logging.info("  FINAL RESULTS: Original SAT vs DSGR (Your Innovation)")
    logging.info("=" * 60)

    if "original_SAT" in results and "DSGR_CoT" in results:
        orig = results["original_SAT"]
        dsgr = results["DSGR_CoT"]

        logging.info(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  指标          原版 SAT          DSGR (创新版)     差值   │
  ├──────────────────────────────────────────────────────────┤
  │  Hit@1         {orig['Hit@1']:>6.2f}%           {dsgr['Hit@1']:>6.2f}%        {dsgr['Hit@1']-orig['Hit@1']:>+.2f}%  │
  │  Hit@3         {orig.get('Hit@3', 0):>6.2f}%           {dsgr.get('Hit@3', 0):>6.2f}%        {dsgr.get('Hit@3', 0)-orig.get('Hit@3', 0):>+.2f}%  │
  │  Hit@10        {orig.get('Hit@10', 0):>6.2f}%           {dsgr.get('Hit@10', 0):>6.2f}%        {dsgr.get('Hit@10', 0)-orig.get('Hit@10', 0):>+.2f}%  │
  │  MRR           {orig['MRR']:>6.2f}%           {dsgr['MRR']:>6.2f}%        {dsgr['MRR']-orig['MRR']:>+.2f}%  │
  │  评估样本       {orig['total']:>5d}             {dsgr['total']:>5d}                │
  └──────────────────────────────────────────────────────────┘
""")
        if dsgr['Hit@1'] > orig['Hit@1']:
            logging.info("  结论: DSGR 创新方法有效，Hit@1 有提升!")
        elif dsgr['Hit@1'] == orig['Hit@1']:
            logging.info("  结论: 两者持平，可能需要更多训练数据或更大模型")
        else:
            logging.info("  结论: 原版暂时更好，建议增大 CoT 训练数据量")
    else:
        for name, r in results.items():
            logging.info(f"  [{name}] Hit@1={r['Hit@1']:.2f}%, MRR={r['MRR']:.2f}%")

    report_path = os.path.join(RUN_DIR, "final_results.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"\n  完整结果保存到: {report_path}")


def main():
    start_time = time.time()
    os.makedirs(RUN_DIR, exist_ok=True)
    cfg_path = save_runtime_snapshot()

    logging.info("DSGR Full Pipeline with Qwen2.5-0.5B")
    logging.info(f"Run ID: {RUN_ID}")
    logging.info(f"Run Dir: {RUN_DIR}")
    logging.info(f"Seed: {GLOBAL_SEED}")
    logging.info(f"Resume: {RESUME_TRAIN}")
    logging.info(f"Config snapshot: {cfg_path}")
    logging.info(f"Model: {QWEN_MODEL}")

    run_stage1()
    run_cot_generation()
    results = run_stage2()
    final_report(results)

    elapsed = time.time() - start_time
    logging.info(f"\n总耗时: {elapsed/3600:.1f} 小时")


# ==================== Main ====================
if __name__ == "__main__":
    main()
