"""
SAT 全流程评估脚本（CPU 版）
评估 Stage 1 (Aligner 对齐质量) 和 Stage 2 (Predictor 链接预测) 的效果。
用法: conda activate sat_cpu && cd SAT && python evaluate_all.py
"""
import os
import sys
import json
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from test_aligner_cpu import (
    CLIPModel, GraphTransformer,
    get_id_map, get_id2text, load_triples, build_graph,
    TAGDataset, tokenize
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

CKPT_DIR = os.path.join("checkpoints_cpu", "FB15k-237N")
DATA_DIR = os.path.join("aligner", "data", "FB15k-237N")
PRED_DATA = os.path.join("predictor", "data_llm_lp", "FB15k-237N")
DEVICE = torch.device("cpu")


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_metric(name, value, fmt=".4f"):
    print(f"  {name:<30} {value:{fmt}}")


# ================================================================
# Part 1: Aligner 评估
# ================================================================
def evaluate_aligner():
    print_header("Stage 1: Aligner 对齐质量评估")

    with open(os.path.join(CKPT_DIR, "config.json")) as f:
        cfg = json.load(f)

    ent2id = get_id_map(os.path.join(DATA_DIR, "mid2id.txt"))
    rel2id = get_id_map(os.path.join(DATA_DIR, "rel2id.txt"))
    id2text = get_id2text(os.path.join(DATA_DIR, "id2text.txt"))
    entity_num, relation_num = len(ent2id), len(rel2id)

    src_all, dst_all, rel_all, _ = load_triples(DATA_DIR, ['train', 'valid', 'test'], ent2id, rel2id)
    graph = build_graph(src_all, dst_all, rel_all, entity_num, relation_num).to(DEVICE)

    print("\n  加载模型...")
    model = CLIPModel(
        entity_num=entity_num, relation_num=relation_num,
        embed_dim=cfg.get('embed_dim', 32), context_length=cfg.get('context_length', 64),
        vocab_size=cfg.get('vocab_size', 49408),
        t_width=cfg.get('transformer_width', 64),
        t_layers=cfg.get('transformer_layers', 2),
        t_heads=cfg.get('transformer_heads', 4),
        gnn_input=cfg.get('gnn_input', 32),
        gnn_output=cfg.get('embed_dim', 32),
        gt_layers=cfg.get('gt_layers', 2),
        gt_head=cfg.get('gt_head', 4),
        neigh_num=cfg.get('neigh_num', 3),
        lr=cfg.get('lr', 1e-4)
    ).to(DEVICE)

    state = torch.load(os.path.join(CKPT_DIR, "aligner_best.pkl"), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print("  模型加载完成")

    # --- 1.1 对齐准确率 (Test Set) ---
    print("\n  [1.1] 对齐准确率 (Alignment Accuracy)")
    _, _, _, pos_tails_test = load_triples(DATA_DIR, ['test'], ent2id, rel2id)
    test_ds = TAGDataset(dict(list(pos_tails_test.items())[:500]), relation_num, 3)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=TAGDataset.collate_fn)

    ctx_len = cfg.get('context_length', 64)
    all_true, all_pred_sg, all_pred_gt, all_pred_tt = [], [], [], []

    with torch.no_grad():
        for src, rel, dst in tqdm(test_loader, desc="  评估中"):
            src_arr, dst_arr = src.numpy(), dst.numpy().reshape(-1)
            src_texts = tokenize([id2text.get(i, "") for i in src_arr], context_length=ctx_len)
            dst_texts = tokenize([id2text.get(j, "") for j in dst_arr], context_length=ctx_len)

            s_graph, s_text, t_text, labels = model(graph, src, rel, dst, src_texts, dst_texts, DEVICE)

            true_labels = labels.numpy().tolist()
            all_true.extend(true_labels)
            all_pred_sg.extend(model.align_pred(s_graph, s_text).numpy().tolist())
            all_pred_gt.extend(model.align_pred(s_graph, t_text).numpy().tolist())
            all_pred_tt.extend(model.align_pred(s_text, t_text).numpy().tolist())

    acc_sg = accuracy_score(all_true, all_pred_sg)
    acc_gt = accuracy_score(all_true, all_pred_gt)
    acc_tt = accuracy_score(all_true, all_pred_tt)
    acc_avg = (acc_sg + acc_gt + acc_tt) / 3

    print_metric("Graph ↔ Source Text", acc_sg)
    print_metric("Graph ↔ Target Text", acc_gt)
    print_metric("Source Text ↔ Target Text", acc_tt)
    print_metric("平均对齐准确率", acc_avg)

    # --- 1.2 嵌入质量分析 ---
    print("\n  [1.2] 嵌入质量分析")

    with torch.no_grad():
        all_ent_embeds = model.gnn(graph)

    norms = all_ent_embeds.norm(dim=-1)
    print_metric("嵌入维度", all_ent_embeds.shape[1], "d")
    print_metric("平均 L2 范数", norms.mean().item())
    print_metric("范数标准差", norms.std().item())
    print_metric("最小范数", norms.min().item())
    print_metric("最大范数", norms.max().item())

    # 有效秩 (effective rank) - 衡量嵌入空间利用率
    emb_np = all_ent_embeds.numpy()
    _, S, _ = np.linalg.svd(emb_np[:2000], full_matrices=False)
    S_norm = S / S.sum()
    entropy = -(S_norm * np.log(S_norm + 1e-10)).sum()
    eff_rank = np.exp(entropy)
    print_metric("有效秩 (Effective Rank)", eff_rank)
    print_metric("  (满秩=" + str(all_ent_embeds.shape[1]) + ")", eff_rank / all_ent_embeds.shape[1], ".1%")

    # --- 1.3 最近邻检索质量 ---
    print("\n  [1.3] 最近邻语义一致性 (随机抽 10 个实体)")

    sample_ids = random.sample(range(entity_num), 10)
    emb_normed = F.normalize(all_ent_embeds, dim=-1)

    for eid in sample_ids:
        query = emb_normed[eid:eid + 1]
        sims = (query @ emb_normed.t()).squeeze()
        topk_vals, topk_ids = sims.topk(4)
        query_text = id2text.get(eid, "???")[:60]
        neighbors = []
        for j in range(1, 4):
            nid = topk_ids[j].item()
            ntext = id2text.get(nid, "???")[:40]
            neighbors.append(f"[{nid}] {ntext} (sim={topk_vals[j]:.3f})")
        print(f"\n  Entity {eid}: {query_text}")
        for n in neighbors:
            print(f"    → {n}")

    return {
        "align_acc_sg": acc_sg, "align_acc_gt": acc_gt, "align_acc_tt": acc_tt,
        "align_acc_avg": acc_avg, "effective_rank": float(eff_rank),
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std())
    }


# ================================================================
# Part 2: Predictor 评估
# ================================================================
def evaluate_predictor():
    print_header("Stage 2: Predictor 链接预测评估")

    pred_test_path = os.path.join(PRED_DATA, "test.json")
    if not os.path.exists(pred_test_path):
        print("  测试数据不存在，跳过 Stage 2 评估")
        return {}

    graph_data_path = os.path.join(CKPT_DIR, "graph_data_all.pt")
    if not os.path.exists(graph_data_path):
        print("  graph_data_all.pt 不存在，跳过 Stage 2 评估")
        return {}

    with open(pred_test_path) as f:
        test_data = json.load(f)

    print(f"  测试数据: {len(test_data)} 条")

    # --- 2.1 数据分布统计 ---
    print("\n  [2.1] 测试数据分布")
    answer_lengths = []
    has_graph = 0
    graph_sizes = []

    for item in test_data:
        answer = item['conversations'][1]['value']
        answer_lengths.append(len(answer.split()))
        if 'graph' in item:
            has_graph += 1
            graph_sizes.append(len(item['graph']['node_list']))

    print_metric("总样本数", len(test_data), "d")
    print_metric("含图样本数", has_graph, "d")
    print_metric("含图比例", has_graph / len(test_data))
    print_metric("答案平均长度(词)", np.mean(answer_lengths))
    if graph_sizes:
        print_metric("子图平均节点数", np.mean(graph_sizes))
        print_metric("子图最大节点数", max(graph_sizes), "d")
        print_metric("子图最小节点数", min(graph_sizes), "d")

    # --- 2.2 用 Graph Encoder 评估子图表示质量 ---
    print("\n  [2.2] 子图表示质量 (Graph Encoder)")

    with open(os.path.join(CKPT_DIR, "config.json")) as f:
        cfg = json.load(f)

    graph_data_all = torch.load(graph_data_path, map_location="cpu", weights_only=False)

    from test_predictor_cpu import GraphEncoder
    encoder = GraphEncoder(
        entity_num=cfg['entity_num'], relation_num=cfg['relation_num'],
        d_in=cfg['gnn_input'], d_out=cfg.get('embed_dim', 32),
        n_layers=cfg['gt_layers'], n_head=cfg['gt_head']
    )
    aligner_state = torch.load(os.path.join(CKPT_DIR, "aligner_best.pkl"),
                               map_location="cpu", weights_only=False)
    gnn_state = {k.replace("gnn.", ""): v for k, v in aligner_state.items() if k.startswith("gnn.")}
    encoder.load_state_dict(gnn_state)
    encoder.eval()

    subgraph_norms = []
    target_node_norms = []
    n_eval = min(200, len(test_data))

    with torch.no_grad():
        for item in tqdm(test_data[:n_eval], desc="  编码子图"):
            if 'graph' not in item:
                continue
            gd = item['graph']
            node_list = gd['node_list']
            edge_index = torch.tensor(gd['edge_index'], dtype=torch.long)
            graph_type = item['id'].split('_')[0]
            node_rep = graph_data_all[graph_type].x[node_list]

            subgraph = Data(
                graph_node=node_rep,
                edge_index=edge_index,
                entity=torch.arange(len(node_list))
            )
            out = encoder(subgraph)
            subgraph_norms.append(out.norm(dim=-1).mean().item())

            target_idx = gd['node_idx']
            if isinstance(target_idx, list):
                target_idx = target_idx[0]
            if target_idx < out.shape[0]:
                target_node_norms.append(out[target_idx].norm().item())

    if subgraph_norms:
        print_metric("子图嵌入平均范数", np.mean(subgraph_norms))
        print_metric("子图嵌入范数标准差", np.std(subgraph_norms))
    if target_node_norms:
        print_metric("目标节点平均范数", np.mean(target_node_norms))

    # --- 2.3 Baseline 评估 (随机/多数类) ---
    print("\n  [2.3] Baseline 对比")

    answers = [item['conversations'][1]['value'] for item in test_data[:1000]]
    unique_answers = list(set(answers))
    from collections import Counter
    answer_counts = Counter(answers)
    most_common = answer_counts.most_common(1)[0]

    majority_acc = most_common[1] / len(answers)
    random_acc = 1.0 / max(len(unique_answers), 1)

    print_metric("唯一答案数", len(unique_answers), "d")
    print_metric("最常见答案", most_common[0][:50], "s")
    print_metric("多数类 Baseline Acc", majority_acc)
    print_metric("随机 Baseline Acc", random_acc)
    print(f"\n  论文报告的 Llama-2-7B 在 FB15k-237N 上:")
    print(f"    Hit@1 ≈ 50-60%")
    print(f"    你需要超过多数类 Baseline ({majority_acc:.1%}) 才算有效")

    return {
        "n_test": len(test_data), "has_graph_ratio": has_graph / len(test_data),
        "avg_answer_len": float(np.mean(answer_lengths)),
        "majority_baseline": majority_acc, "random_baseline": random_acc,
        "unique_answers": len(unique_answers)
    }


# ================================================================
# Part 3: 生成评估报告
# ================================================================
def generate_report(aligner_metrics, predictor_metrics):
    print_header("综合评估报告")

    print("\n  ┌─ Stage 1: Aligner ─────────────────────────┐")
    if aligner_metrics:
        acc = aligner_metrics['align_acc_avg']
        rank = aligner_metrics['effective_rank']
        if acc > 0.5:
            verdict1 = "良好 - 对齐效果显著"
        elif acc > 0.2:
            verdict1 = "一般 - 需增加训练轮次/数据量"
        else:
            verdict1 = "较弱 - 参数太小或训练不足（dry-run 预期）"
        print(f"  │ 对齐准确率:    {acc:.1%}")
        print(f"  │ 有效秩:        {rank:.1f}/{aligner_metrics.get('embed_dim', 32)}")
        print(f"  │ 判定:          {verdict1}")
    print("  └────────────────────────────────────────────┘")

    print("\n  ┌─ Stage 2: Predictor ────────────────────────┐")
    if predictor_metrics:
        maj = predictor_metrics['majority_baseline']
        n_ans = predictor_metrics['unique_answers']
        print(f"  │ 测试样本:      {predictor_metrics['n_test']}")
        print(f"  │ 唯一答案数:    {n_ans}")
        print(f"  │ 多数类Baseline: {maj:.1%}")
        print(f"  │ 目标 (论文):   Hit@1 ≈ 50-60%")
        if predictor_metrics.get('has_graph_ratio', 0) > 0.9:
            print(f"  │ 图数据覆盖:    完整")
        else:
            print(f"  │ 图数据覆盖:    {predictor_metrics['has_graph_ratio']:.0%}")
    print("  └────────────────────────────────────────────┘")

    print("\n  提升建议:")
    print("  1. Stage 1: 增大模型维度 (embed_dim=128, t_layers=12)")
    print("     当前 dry-run 用小参数，对齐能力有限")
    print("  2. Stage 1: 训练更多轮次 (100 epochs) 和全量数据")
    print("  3. Stage 2: 使用预训练 LLM (Llama-2-7B 或 TinyLlama-1.1B)")
    print("     当前用随机初始化 Llama，无语言理解能力")
    print("  4. 在 GPU 服务器上训练可获得论文级效果")

    all_metrics = {**aligner_metrics, **predictor_metrics}
    report_path = os.path.join(CKPT_DIR, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  评估指标已保存到: {report_path}")


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print_header("SAT 全流程评估")
    print(f"  Checkpoint 目录: {CKPT_DIR}")
    print(f"  数据目录:        {DATA_DIR}")
    print(f"  设备:            {DEVICE}")

    aligner_metrics = evaluate_aligner()
    predictor_metrics = evaluate_predictor()
    generate_report(aligner_metrics, predictor_metrics)
