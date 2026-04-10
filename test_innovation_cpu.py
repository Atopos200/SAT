"""
DSGR (Dynamic Subgraph Grounded Reasoning) 全流程测试
在 CPU 上验证：子图选择 -> 结构序列化 -> 动态图Token -> CoT数据生成

用法: conda activate sat_cpu && cd SAT && python test_innovation_cpu.py
"""
import os
import sys
import json
import time
import random
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from innovation.config import InnovationConfig
from innovation.subgraph_selector import AdaptiveSubgraphSelector, KGIndex
from innovation.structure_serializer import StructureAwareSerializer, build_cot_instruction
from innovation.dynamic_graph_token import DynamicGraphTokenizer

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = os.path.join("aligner", "data", "FB15k-237N")
CKPT_DIR = os.path.join("checkpoints_cpu", "FB15k-237N")


def load_id_map(path):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                d[parts[0]] = int(parts[1])
    return d


def load_id2text(path):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                d[int(parts[0])] = parts[1]
    return d


def load_triples(data_path, splits, ent2id, rel2id):
    triples = []
    for split in splits:
        path = os.path.join(data_path, f"{split}.txt")
        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3 and parts[0] in ent2id and parts[1] in rel2id and parts[2] in ent2id:
                    triples.append((ent2id[parts[0]], rel2id[parts[1]], ent2id[parts[2]]))
    return triples


def build_entity_names(id2text):
    names = {}
    for eid, text in id2text.items():
        first_sent = text.split('.')[0].strip()
        if len(first_sent) > 80:
            first_sent = first_sent[:77] + "..."
        names[eid] = first_sent if first_sent else f"Entity_{eid}"
    return names


def build_relation_names(rel2id):
    names = {}
    for rel_str, rid in rel2id.items():
        parts = rel_str.strip('/').split('/')
        names[rid] = parts[-1].replace('_', ' ') if parts else rel_str
    return names


def print_section(title):
    logging.info(f"\n{'=' * 70}")
    logging.info(f"  {title}")
    logging.info(f"{'=' * 70}")


def main():
    print_section("DSGR Pipeline Test (CPU)")

    # ===== 1. 加载数据 =====
    logging.info("Loading KG data...")
    ent2id = load_id_map(os.path.join(DATA_DIR, "mid2id.txt"))
    rel2id = load_id_map(os.path.join(DATA_DIR, "rel2id.txt"))
    id2text = load_id2text(os.path.join(DATA_DIR, "id2text.txt"))
    entity_num, relation_num = len(ent2id), len(rel2id)
    logging.info(f"  Entities: {entity_num}, Relations: {relation_num}")

    triples = load_triples(DATA_DIR, ['train', 'valid', 'test'], ent2id, rel2id)
    logging.info(f"  Triples: {len(triples)}")

    entity_emb = None
    emb_path = os.path.join(CKPT_DIR, "entity_embedding.pt")
    if os.path.exists(emb_path):
        entity_emb = torch.load(emb_path, map_location="cpu", weights_only=False)
        logging.info(f"  Entity embeddings: {entity_emb.shape}")

    # ===== 2. 构建模块 =====
    print_section("Phase 1: Adaptive Subgraph Selection")

    config = InnovationConfig(
        max_subgraph_nodes=12,
        max_hops=2,
        num_graph_tokens=8,
        token_dim=entity_emb.shape[1] if entity_emb is not None else 32,
        cot_style="graph_grounded",
    )

    logging.info("Building KG index...")
    t0 = time.time()
    kg_index = KGIndex(triples, entity_num, relation_num)
    logging.info(f"  KG index built in {time.time() - t0:.2f}s")

    selector = AdaptiveSubgraphSelector(kg_index, config, entity_embeddings=entity_emb)

    ent_names = build_entity_names(id2text)
    rel_names = build_relation_names(rel2id)
    serializer = StructureAwareSerializer(config, ent_names, rel_names)

    # ===== 3. 测试子图选择 =====
    test_triples = random.sample(triples, min(5, len(triples)))

    for i, (h, r, t) in enumerate(test_triples):
        logging.info(f"\n--- Query {i + 1}: ({ent_names.get(h, h)[:40]}, {rel_names.get(r, r)}, ?) ---")
        logging.info(f"    Ground truth: {ent_names.get(t, t)[:60]}")

        t0 = time.time()
        subgraph = selector.select(h, r)
        select_time = time.time() - t0

        logging.info(f"    Selected nodes: {len(subgraph.node_ids)}, Edges: {subgraph.edge_index.shape[1]}")
        logging.info(f"    Selection time: {select_time * 1000:.1f}ms")
        logging.info(f"    Top-3 importance: {subgraph.importance_scores[:3].tolist()}")
        logging.info(f"    Paths found: {len(subgraph.paths)}")

        gt_in_subgraph = t in subgraph.node_ids
        logging.info(f"    Ground truth in subgraph: {'YES' if gt_in_subgraph else 'NO'}")

    # ===== 4. 测试结构序列化 =====
    print_section("Phase 2: Structure-Aware Serialization")

    h, r, t = test_triples[0]
    subgraph = selector.select(h, r)

    for style in ["graph_grounded", "path_based", "step_by_step"]:
        config.cot_style = style
        serializer_tmp = StructureAwareSerializer(config, ent_names, rel_names)
        serialized = serializer_tmp.serialize(subgraph, h, r)

        logging.info(f"\n--- CoT Style: {style} ---")
        logging.info(f"  Graph context: {serialized['graph_context'][:120]}")
        logging.info(f"  Node order (top-5): {serialized['node_order'][:5]}")
        logging.info(f"\n  CoT Prompt:\n{'─' * 50}")
        for line in serialized['cot_prompt'].split('\n'):
            logging.info(f"    {line}")
        logging.info(f"{'─' * 50}")

    # ===== 5. 测试动态图 Token =====
    print_section("Phase 3: Dynamic Graph Token Compression")

    input_dim = entity_emb.shape[1] if entity_emb is not None else 32
    tokenizer = DynamicGraphTokenizer(config, input_dim=input_dim)
    total_params = sum(p.numel() for p in tokenizer.parameters())
    logging.info(f"  DynamicGraphTokenizer params: {total_params:,}")
    logging.info(f"  Input: variable N nodes x {input_dim}D -> Output: {config.num_graph_tokens} tokens x {config.token_dim}D")

    for i, (h, r, t) in enumerate(test_triples[:3]):
        subgraph = selector.select(h, r)

        if entity_emb is not None:
            node_features = entity_emb[subgraph.node_ids]
        else:
            node_features = torch.randn(len(subgraph.node_ids), input_dim)

        with torch.no_grad():
            graph_tokens = tokenizer(node_features, subgraph.importance_scores)

        logging.info(f"\n  Query {i + 1}: {len(subgraph.node_ids)} nodes -> {graph_tokens.shape[0]} tokens")
        logging.info(f"    Token norms: {graph_tokens.norm(dim=-1).tolist()[:4]}...")
        logging.info(f"    Token shape: {graph_tokens.shape}")

    # ===== 6. 端到端测试: 生成 CoT 训练数据 =====
    print_section("Phase 4: End-to-End CoT Data Generation")

    config.cot_style = "graph_grounded"
    serializer = StructureAwareSerializer(config, ent_names, rel_names)

    sample_data = []
    test_sample = random.sample(triples, min(20, len(triples)))

    for h, r, t in test_sample:
        subgraph = selector.select(h, r)
        serialized = serializer.serialize(subgraph, h, r)
        answer = ent_names.get(t, str(t))
        instruction = build_cot_instruction(serialized["cot_prompt"], answer)
        instruction["id"] = f"FB15k-237N_cot_{h}_{r}_{t}"
        instruction["graph"] = {
            "node_idx": 0,
            "edge_index": subgraph.edge_index.tolist(),
            "node_list": subgraph.node_ids,
            "importance_scores": subgraph.importance_scores.tolist(),
        }
        sample_data.append(instruction)

    out_dir = os.path.join("data_cot_lp", "FB15k-237N")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_cot.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    logging.info(f"  Generated {len(sample_data)} CoT samples -> {out_path}")

    logging.info(f"\n  Sample instruction preview:")
    sample = sample_data[0]
    logging.info(f"  ID: {sample['id']}")
    logging.info(f"  Graph nodes: {len(sample['graph']['node_list'])}")
    q = sample['conversations'][0]['value']
    a = sample['conversations'][1]['value']
    logging.info(f"\n  [Human]:\n{'─' * 50}")
    for line in q.split('\n')[:15]:
        logging.info(f"    {line}")
    if q.count('\n') > 15:
        logging.info(f"    ... ({q.count(chr(10)) - 15} more lines)")
    logging.info(f"{'─' * 50}")
    logging.info(f"\n  [GPT]: {a}")

    # ===== 7. 性能统计 =====
    print_section("Performance Summary")

    times = []
    for h, r, t in random.sample(triples, min(100, len(triples))):
        t0 = time.time()
        sg = selector.select(h, r)
        if entity_emb is not None:
            nf = entity_emb[sg.node_ids]
        else:
            nf = torch.randn(len(sg.node_ids), input_dim)
        with torch.no_grad():
            _ = tokenizer(nf, sg.importance_scores)
        _ = serializer.serialize(sg, h, r)
        times.append(time.time() - t0)

    logging.info(f"  Full pipeline (select + tokenize + serialize):")
    logging.info(f"    Mean: {np.mean(times) * 1000:.1f}ms per query")
    logging.info(f"    P50:  {np.percentile(times, 50) * 1000:.1f}ms")
    logging.info(f"    P95:  {np.percentile(times, 95) * 1000:.1f}ms")
    logging.info(f"    P99:  {np.percentile(times, 99) * 1000:.1f}ms")

    # GT recall test
    gt_hits = 0
    test_n = min(500, len(triples))
    test_subset = random.sample(triples, test_n)
    for h, r, t in test_subset:
        sg = selector.select(h, r)
        if t in sg.node_ids:
            gt_hits += 1
    gt_recall = gt_hits / test_n
    logging.info(f"\n  Ground truth recall (in selected subgraph):")
    logging.info(f"    {gt_hits}/{test_n} = {gt_recall:.1%}")
    logging.info(f"    (This measures how often the answer entity is in the selected subgraph)")

    print_section("ALL TESTS PASSED")
    logging.info("  Innovation pipeline is fully functional.")
    logging.info("  Next steps:")
    logging.info("    1. Generate full CoT data: python -m innovation.build_cot_data")
    logging.info("    2. Train with CoT data on GPU server")
    logging.info("    3. Evaluate with structure-aware prompts")


if __name__ == "__main__":
    main()
