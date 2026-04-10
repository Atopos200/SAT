"""
CoT 数据构造脚本
读取原始 KG 数据 + 原始指令数据，用 Adaptive Subgraph Selector + Structure-Aware Serializer
生成带推理链的新指令数据。

用法: cd SAT && python -m innovation.build_cot_data --output_dir ./data_cot_lp/FB15k-237N
"""
import os
import sys
import json
import argparse
import logging
import re
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm

from innovation.config import InnovationConfig
from innovation.subgraph_selector import AdaptiveSubgraphSelector, KGIndex
from innovation.structure_serializer import StructureAwareSerializer, build_cot_instruction
from dsgr.data.manifest import init_split_stats, add_error, add_note

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def load_id_map(path: str) -> Dict[str, int]:
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                d[parts[0]] = int(parts[1])
    return d


def load_id2text(path: str) -> Dict[int, str]:
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                d[int(parts[0])] = parts[1]
    return d


def load_triples(data_path: str, splits: List[str], ent2id: Dict, rel2id: Dict) -> List[Tuple[int, int, int]]:
    triples = []
    for split in splits:
        path = os.path.join(data_path, f"{split}.txt")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    s, r, d = parts
                    if s in ent2id and r in rel2id and d in ent2id:
                        triples.append((ent2id[s], rel2id[r], ent2id[d]))
    return triples


def build_entity_names(id2text: Dict[int, str]) -> Dict[int, str]:
    """从描述文本中提取简短实体名（取第一句）"""
    names = {}
    for eid, text in id2text.items():
        first_sent = text.split('.')[0].strip()
        if len(first_sent) > 100:
            first_sent = first_sent[:97] + "..."
        names[eid] = first_sent if first_sent else f"Entity_{eid}"
    return names


def build_relation_names(rel2id: Dict[str, int]) -> Dict[int, str]:
    """将 Freebase 风格的关系路径转为可读名称"""
    names = {}
    for rel_str, rid in rel2id.items():
        parts = rel_str.strip('/').split('/')
        readable = parts[-1].replace('_', ' ') if parts else rel_str
        names[rid] = readable
    return names


def simple_tokenize(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2]


def build_relation_tail_prototypes(
    triples: List[Tuple[int, int, int]],
    id2text: Dict[int, str],
    topk: int = 8,
) -> Dict[int, List[str]]:
    rel_words = defaultdict(list)
    for _, r, t in triples:
        text = id2text.get(t, "")
        rel_words[r].extend(simple_tokenize(text))

    prototypes = {}
    for r, words in rel_words.items():
        if not words:
            continue
        prototypes[r] = [w for w, _ in Counter(words).most_common(topk)]
    return prototypes


def process_split(split: str, config: InnovationConfig, selector: AdaptiveSubgraphSelector,
                  serializer: StructureAwareSerializer, original_data_path: str,
                  ent2id: Dict, rel2id: Dict, triples: List[Tuple[int, int, int]]) -> Tuple[List[Dict], Dict]:
    """处理单个 split（train/valid/test），生成 CoT 指令数据"""

    original_file = os.path.join(original_data_path, f"{split}.json")
    if os.path.exists(original_file):
        logging.info(f"  加载原始指令数据: {original_file}")
        with open(original_file) as f:
            original_data = json.load(f)
        return _process_from_instruction(split, original_data, config, selector, serializer)

    triple_file = os.path.join(config.data_path, f"{split}.txt")
    if os.path.exists(triple_file):
        logging.info(f"  从三元组文件生成: {triple_file}")
        return _process_from_triples(split, triple_file, config, selector, serializer, ent2id, rel2id)

    stats = init_split_stats(split, "none")
    stats["notes"].append("No source file found for this split.")
    return [], stats


def _process_from_instruction(split: str, data: List[Dict], config: InnovationConfig,
                              selector: AdaptiveSubgraphSelector,
                              serializer: StructureAwareSerializer) -> Tuple[List[Dict], Dict]:
    """从已有指令数据增强为 CoT 版本"""
    stats = init_split_stats(split, "instruction_json")
    max_samples = getattr(config, '_max_samples', -1)
    if max_samples > 0:
        data = data[:max_samples]
    stats["total_input"] = len(data)
    results = []
    for item in tqdm(data, desc="  生成 CoT 数据"):
        if 'graph' not in item:
            results.append(item)
            stats["fallback"] += 1
            continue

        graph_dict = item['graph']
        node_list = graph_dict['node_list']
        original_answer = item['conversations'][1]['value']

        conv_text = item['conversations'][0]['value']
        head_node = node_list[0] if node_list else 0

        rel_id = 0
        item_id = item.get('id', '')
        parts = item_id.split('_')
        if len(parts) > 1:
            try:
                rel_id = int(parts[1]) % 237
            except (ValueError, IndexError):
                add_note(stats, f"Relation parse fallback for item_id={item_id}")
                pass

        try:
            subgraph = selector.select(head_node, rel_id, max_nodes=config.max_subgraph_nodes)
            serialized = serializer.serialize(subgraph, head_node, rel_id)

            prompt_text = serialized.get("structured_prompt") or serialized["cot_prompt"]
            new_instruction = build_cot_instruction(
                prompt_text,
                original_answer,
                graph_summary=serialized.get("graph_summary", ""),
            )

            new_item = {
                "id": item_id + "_cot",
                "graph": {
                    "node_idx": graph_dict.get('node_idx', 0),
                    "edge_index": subgraph.edge_index.tolist(),
                    "node_list": subgraph.node_ids,
                    "importance_scores": subgraph.importance_scores.tolist(),
                    "paths": subgraph.paths,
                },
                "conversations": new_instruction["conversations"],
                "graph_summary": serialized.get("graph_summary", ""),
                "structured_prompt": prompt_text,
                "original_graph": graph_dict,
            }
            results.append(new_item)
            stats["success"] += 1

        except Exception as e:
            results.append(item)
            stats["fallback"] += 1
            add_error(stats, e)

    stats["total_output"] = len(results)
    return results, stats


def _process_from_triples(split: str, triple_file: str, config: InnovationConfig,
                          selector: AdaptiveSubgraphSelector,
                          serializer: StructureAwareSerializer,
                          ent2id: Dict, rel2id: Dict) -> Tuple[List[Dict], Dict]:
    """从三元组文件直接生成 CoT 数据"""
    stats = init_split_stats(split, "triple_txt")
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    results = []

    with open(triple_file) as f:
        lines = f.readlines()

    max_samples = getattr(config, '_max_samples', -1)
    if max_samples > 0:
        lines = lines[:max_samples]
    stats["total_input"] = len(lines)
    for i, line in enumerate(tqdm(lines, desc="  生成 CoT 数据")):
        parts = line.strip().split('\t')
        if len(parts) != 3:
            stats["skipped"] += 1
            continue
        s, r, d = parts
        if s not in ent2id or r not in rel2id or d not in ent2id:
            stats["skipped"] += 1
            continue

        h_id, r_id, t_id = ent2id[s], rel2id[r], ent2id[d]

        try:
            subgraph = selector.select(h_id, r_id, max_nodes=config.max_subgraph_nodes)
            serialized = serializer.serialize(subgraph, h_id, r_id)

            answer = serializer.get_entity_name(t_id)
            prompt_text = serialized.get("structured_prompt") or serialized["cot_prompt"]
            new_item = build_cot_instruction(
                prompt_text,
                answer,
                graph_summary=serialized.get("graph_summary", ""),
            )
            new_item["id"] = f"FB15k-237N_{i}_cot"
            new_item["graph"] = {
                "node_idx": 0,
                "edge_index": subgraph.edge_index.tolist(),
                "node_list": subgraph.node_ids,
                "importance_scores": subgraph.importance_scores.tolist(),
                "paths": subgraph.paths,
            }
            new_item["graph_summary"] = serialized.get("graph_summary", "")
            new_item["structured_prompt"] = prompt_text
            results.append(new_item)
            stats["success"] += 1

        except Exception as e:
            stats["skipped"] += 1
            add_error(stats, e)
            continue

    stats["total_output"] = len(results)
    return results, stats


def main():
    parser = argparse.ArgumentParser(description="Build CoT instruction data for DSGR")
    parser.add_argument("--data_name", default="FB15k-237N")
    parser.add_argument("--data_path", default="aligner/data/FB15k-237N")
    parser.add_argument("--original_data_path", default="predictor/data_llm_lp/FB15k-237N")
    parser.add_argument("--output_dir", default="data_cot_lp/FB15k-237N")
    parser.add_argument("--embedding_path", default=None, help="Path to entity_embedding.pt")
    parser.add_argument("--max_subgraph_nodes", type=int, default=16)
    parser.add_argument("--cot_style", default="graph_grounded", choices=["graph_grounded", "path_based", "step_by_step"])
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 = all")
    parser.add_argument("--selector_mode", default="rule_based", choices=["rule_based", "learned"])
    parser.add_argument("--selector_debug_log", action="store_true")
    parser.add_argument("--selector_debug_path", default="results_qwen/selector_debug.jsonl")
    args = parser.parse_args()

    config = InnovationConfig(
        data_path=args.data_path,
        max_subgraph_nodes=args.max_subgraph_nodes,
        cot_style=args.cot_style,
    )
    config._max_samples = args.max_samples
    config.selector_debug_log = bool(args.selector_debug_log)
    config.selector_debug_path = args.selector_debug_path

    logging.info("Loading KG data...")
    ent2id = load_id_map(os.path.join(args.data_path, "mid2id.txt"))
    rel2id = load_id_map(os.path.join(args.data_path, "rel2id.txt"))
    id2text = load_id2text(os.path.join(args.data_path, "id2text.txt"))
    entity_num, relation_num = len(ent2id), len(rel2id)
    logging.info(f"  Entities: {entity_num}, Relations: {relation_num}")

    logging.info("Loading triples and building KG index...")
    triples = load_triples(args.data_path, ['train', 'valid', 'test'], ent2id, rel2id)
    kg_index = KGIndex(triples, entity_num, relation_num)
    logging.info(f"  Total triples: {len(triples)}")

    entity_emb = None
    if args.embedding_path and os.path.exists(args.embedding_path):
        logging.info(f"Loading entity embeddings from {args.embedding_path}...")
        entity_emb = torch.load(args.embedding_path, map_location="cpu", weights_only=False)

    relation_tail_prototypes = build_relation_tail_prototypes(
        triples=triples,
        id2text=id2text,
        topk=config.prototype_topk,
    )

    logging.info("Building selector and serializer...")
    selector = AdaptiveSubgraphSelector(
        kg_index,
        config,
        entity_embeddings=entity_emb,
        mode=args.selector_mode,
        entity_texts=id2text,
        relation_tail_prototypes=relation_tail_prototypes,
    )

    ent_names = build_entity_names(id2text)
    rel_names = build_relation_names(rel2id)
    serializer = StructureAwareSerializer(config, ent_names, rel_names)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "config": {
            "max_subgraph_nodes": config.max_subgraph_nodes,
            "max_hops": config.max_hops,
            "expand_top_per_hop": config.expand_top_per_hop,
            "relation_weight": config.relation_weight,
            "structure_weight": config.structure_weight,
            "embedding_weight": config.embedding_weight,
            "path_weight": config.path_weight,
            "type_weight": config.type_weight,
            "role_weight": config.role_weight,
            "redundancy_weight": config.redundancy_weight,
            "cot_style": config.cot_style,
            "max_paths_in_prompt": config.max_paths_in_prompt,
            "max_evidence_items": config.max_evidence_items,
        },
        "kg": {
            "entity_num": entity_num,
            "relation_num": relation_num,
            "triple_num": len(triples),
        },
        "retrieval": {
            "relation_tail_prototype_relations": len(relation_tail_prototypes),
            "prototype_topk": config.prototype_topk,
        },
        "splits": {},
    }

    for split in ['train', 'valid', 'test']:
        logging.info(f"\nProcessing {split}...")
        data, stats = process_split(split, config, selector, serializer,
                                    args.original_data_path, ent2id, rel2id, triples)

        if args.max_samples > 0:
            data = data[:args.max_samples]
            stats["total_output"] = len(data)

        out_path = os.path.join(args.output_dir, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"  Saved {len(data)} items to {out_path}")
        logging.info(
            "  Stats: input=%d, output=%d, success=%d, fallback=%d, skipped=%d, errors=%s",
            stats["total_input"], stats["total_output"], stats["success"],
            stats["fallback"], stats["skipped"], stats["error_types"]
        )
        manifest["splits"][split] = stats

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logging.info(f"\nManifest saved: {manifest_path}")

    logging.info("\nDone!")


if __name__ == "__main__":
    main()
