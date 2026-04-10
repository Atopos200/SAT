"""
Adaptive Subgraph Selector
根据查询 (h, r, ?) 动态选择最相关的子图节点和边。

核心思路：
  1. 从 anchor entity 出发，扩展 k-hop 邻居
  2. 用三维评分（关系相关性 + 结构重要性 + 嵌入相似度）为每个邻居打分
  3. 选 top-K 节点构建 induced subgraph
  4. 输出带重要性排序的子图结构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from innovation.config import InnovationConfig


@dataclass
class SelectedSubgraph:
    """子图选择结果"""
    node_ids: List[int]
    edge_index: torch.LongTensor
    edge_types: List[int]
    importance_scores: torch.FloatTensor
    anchor_idx: int
    paths: List[List[Tuple[int, int, int]]]  # [(src, rel, dst), ...]


class KGIndex:
    """知识图谱邻接索引，支持快速邻居查询"""

    def __init__(self, triples: List[Tuple[int, int, int]], entity_num: int, relation_num: int):
        self.entity_num = entity_num
        self.relation_num = relation_num

        self.head_to_triples = defaultdict(list)
        self.tail_to_triples = defaultdict(list)
        self.relation_cooccur = defaultdict(lambda: defaultdict(int))

        for h, r, t in triples:
            self.head_to_triples[h].append((h, r, t))
            self.tail_to_triples[t].append((h, r, t))

        for h in self.head_to_triples:
            rels = [r for _, r, _ in self.head_to_triples[h]]
            for i, r1 in enumerate(rels):
                for r2 in rels[i + 1:]:
                    self.relation_cooccur[r1][r2] += 1
                    self.relation_cooccur[r2][r1] += 1

    def get_neighbors(self, entity: int) -> List[Tuple[int, int, int]]:
        out_edges = self.head_to_triples.get(entity, [])
        in_edges = [(t, r + self.relation_num, h) for h, r, t in self.tail_to_triples.get(entity, [])]
        return out_edges + in_edges

    def get_relation_similarity(self, r1: int, r2: int) -> float:
        r1_base = r1 % self.relation_num
        r2_base = r2 % self.relation_num
        if r1_base == r2_base:
            return 1.0
        co = self.relation_cooccur[r1_base].get(r2_base, 0)
        total_r1 = sum(self.relation_cooccur[r1_base].values()) + 1
        return co / total_r1


class NeighborScorer(nn.Module):
    """可学习的邻居评分网络"""

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, anchor_emb: torch.Tensor, neighbor_emb: torch.Tensor,
                relation_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([anchor_emb, neighbor_emb, relation_emb], dim=-1)
        return self.scorer(combined).squeeze(-1)


class AdaptiveSubgraphSelector:
    """
    自适应子图选择器

    融合三种信号为邻居打分：
      - 关系相关性: query relation 与邻居连接 relation 的共现频率
      - 结构重要性: 节点度数 + 桥接性
      - 嵌入相似度: 实体嵌入空间中的余弦相似度

    支持两种模式：
      - rule_based: 纯规则打分，不需要训练
      - learned: 用 NeighborScorer 网络打分
    """

    def __init__(self, kg_index: KGIndex, config: InnovationConfig,
                 entity_embeddings: Optional[torch.Tensor] = None,
                 relation_embeddings: Optional[torch.Tensor] = None,
                 mode: str = "rule_based",
                 entity_texts: Optional[Dict[int, str]] = None,
                 relation_tail_prototypes: Optional[Dict[int, List[str]]] = None):
        self.kg = kg_index
        self.config = config
        self.entity_emb = entity_embeddings
        self.relation_emb = relation_embeddings
        self.mode = mode
        self.entity_texts = entity_texts or {}
        self.relation_tail_prototypes = relation_tail_prototypes or {}

        if entity_embeddings is not None:
            self.emb_normed = F.normalize(entity_embeddings, dim=-1)
        else:
            self.emb_normed = None
        if relation_embeddings is not None:
            self.rel_normed = F.normalize(relation_embeddings, dim=-1)
        else:
            self.rel_normed = None

        if mode == "learned" and entity_embeddings is not None:
            embed_dim = entity_embeddings.shape[1]
            self.scorer_net = NeighborScorer(embed_dim, config.scorer_hidden_dim)
            self.scorer_net.eval()
        else:
            self.scorer_net = None

        self._degree_cache = {}
        self._build_degree_cache()
        self._max_degree = max(self._degree_cache.values()) if self._degree_cache else 1
        self._debug_records = 0
        self._token_cache = {}

    def _build_degree_cache(self):
        for ent in range(self.kg.entity_num):
            self._degree_cache[ent] = len(self.kg.get_neighbors(ent))

    def select(self, head: int, relation: int, max_nodes: Optional[int] = None) -> SelectedSubgraph:
        max_nodes = max_nodes or self.config.max_subgraph_nodes
        candidates = self._expand_neighbors(head, relation, self.config.max_hops)

        scored = []
        for node_id, hop, via_relations in candidates:
            if node_id == head:
                continue
            detail = self._score_candidate(head, relation, node_id, hop, via_relations)
            scored.append({
                "node_id": node_id,
                "score": detail["final_score"],
                "hop": hop,
                "via_relations": via_relations,
                "score_detail": detail,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        selected_nodes = [head]
        selected_meta = {head: {"score": 1.0, "hop": 0, "via": []}}
        budget = max(max_nodes - 1, 0)
        hop_quota = self._build_hop_quota(budget)
        hop_used = defaultdict(int)
        rel_used = defaultdict(int)
        score_threshold = self._compute_score_threshold(scored)

        for row in scored:
            node_id = row["node_id"]
            score = row["score"]
            hop = row["hop"]
            via_rels = row["via_relations"]
            if len(selected_nodes) >= max_nodes:
                break
            redundancy_penalty = self._redundancy_penalty(node_id, selected_nodes)
            adjusted_score = score - redundancy_penalty
            if adjusted_score < score_threshold:
                continue
            if self.config.use_hop_quota and hop_quota:
                quota = hop_quota.get(hop, 0)
                if quota <= 0 or hop_used[hop] >= quota:
                    continue
            rel_key = via_rels[0] if via_rels else -1
            if self.config.max_per_relation > 0 and rel_used[rel_key] >= self.config.max_per_relation:
                continue
            if node_id not in selected_meta:
                selected_nodes.append(node_id)
                selected_meta[node_id] = {
                    "score": adjusted_score,
                    "base_score": score,
                    "redundancy_penalty": redundancy_penalty,
                    "hop": hop,
                    "via": via_rels,
                }
                hop_used[hop] += 1
                rel_used[rel_key] += 1

        node_set = set(selected_nodes)
        local_id = {nid: i for i, nid in enumerate(selected_nodes)}
        edges_src, edges_dst, edge_types = [], [], []

        for nid in selected_nodes:
            for h, r, t in self.kg.get_neighbors(nid):
                if t in node_set and h in node_set:
                    src_local = local_id[h] if h in local_id else local_id.get(nid)
                    dst_local = local_id[t] if t in local_id else local_id.get(nid)
                    if src_local is not None and dst_local is not None:
                        edges_src.append(src_local)
                        edges_dst.append(dst_local)
                        edge_types.append(r)

        if not edges_src:
            edges_src, edges_dst, edge_types = [0], [0], [0]

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        scores = torch.tensor([selected_meta[n]["score"] for n in selected_nodes], dtype=torch.float)

        paths = self._extract_key_paths(head, relation, selected_nodes, selected_meta)
        self._maybe_dump_debug(
            head, relation, scored, selected_nodes, selected_meta, hop_quota, score_threshold
        )

        return SelectedSubgraph(
            node_ids=selected_nodes,
            edge_index=edge_index,
            edge_types=edge_types,
            importance_scores=scores,
            anchor_idx=0,
            paths=paths,
        )

    def _expand_neighbors(self, root: int, query_rel: int, max_hops: int) -> List[Tuple[int, int, List[int]]]:
        visited = {root}
        frontier = [(root, 0, [])]
        result = []

        for hop in range(max_hops):
            next_frontier = []
            for node, cur_hop, via_rels in frontier:
                for h, r, t in self.kg.get_neighbors(node):
                    target = t if t != node else h
                    if target not in visited:
                        visited.add(target)
                        new_via = via_rels + [r]
                        result.append((target, cur_hop + 1, new_via))
                        next_frontier.append((target, cur_hop + 1, new_via))
            if self.config.expand_top_per_hop > 0 and len(next_frontier) > self.config.expand_top_per_hop:
                next_frontier = sorted(
                    next_frontier,
                    key=lambda x: self._frontier_priority(root, query_rel, x[0], x[1], x[2]),
                    reverse=True,
                )[:self.config.expand_top_per_hop]
            frontier = next_frontier

        return result

    def _frontier_priority(self, head: int, query_rel: int, candidate: int, hop: int, via_relations: List[int]) -> float:
        rel_score = 0.0
        if via_relations:
            rel_score = max(self.kg.get_relation_similarity(query_rel, r) for r in via_relations)
        emb_score = 0.0
        if self.emb_normed is not None and head < len(self.emb_normed) and candidate < len(self.emb_normed):
            emb_score = float(max(0.0, self.emb_normed[head] @ self.emb_normed[candidate]))
        hop_decay = 1.0 / (1.0 + hop)
        return 0.7 * rel_score + 0.3 * emb_score * hop_decay

    def _score_candidate(self, head: int, query_rel: int, candidate: int,
                         hop: int, via_relations: List[int]) -> Dict[str, float]:
        rel_score = 0.0
        if via_relations:
            rel_scores = [self.kg.get_relation_similarity(query_rel, r) for r in via_relations]
            rel_score = max(rel_scores)

        degree = self._degree_cache.get(candidate, 0)
        struct_score = np.log1p(degree) / np.log1p(self._max_degree)
        hop_decay = 1.0 / (1.0 + hop)
        struct_score *= hop_decay

        emb_score = 0.0
        if self.emb_normed is not None and head < len(self.emb_normed) and candidate < len(self.emb_normed):
            emb_score = float(self.emb_normed[head] @ self.emb_normed[candidate])
            emb_score = max(0.0, emb_score)
        rel_cond_emb_score = emb_score
        if self.config.relation_conditioned_embedding:
            rel_cond_emb_score = self._relation_conditioned_embedding_score(
                head=head, query_rel=query_rel, candidate=candidate, rel_score=rel_score, fallback_emb=emb_score
            )
        path_score = self._path_support_score(query_rel, via_relations) if self.config.enable_path_support else 0.0
        type_score = self._type_consistency_score(query_rel, candidate) if self.config.enable_type_consistency else 0.0
        role_score = self._role_score(query_rel, candidate) if self.config.enable_role_score else 0.0

        learned_score = 0.0
        learned_used = 0.0
        if self.mode == "learned" and self.scorer_net is not None and self.emb_normed is not None:
            if head < len(self.emb_normed) and candidate < len(self.emb_normed):
                head_vec = self.emb_normed[head].unsqueeze(0)
                cand_vec = self.emb_normed[candidate].unsqueeze(0)
                if self.rel_normed is not None and len(self.rel_normed) > 0:
                    rel_base = query_rel % len(self.rel_normed)
                    rel_vec = self.rel_normed[rel_base].unsqueeze(0)
                else:
                    rel_vec = torch.zeros_like(head_vec)
                with torch.no_grad():
                    learned_raw = self.scorer_net(head_vec, cand_vec, rel_vec)
                learned_score = float(torch.sigmoid(learned_raw).item())
                learned_used = 1.0

        w = self.config
        final = (w.relation_weight * rel_score +
                 w.structure_weight * struct_score +
                 w.embedding_weight * rel_cond_emb_score +
                 w.path_weight * path_score +
                 w.type_weight * type_score +
                 w.role_weight * role_score)
        if learned_used > 0 and w.learned_score_weight > 0:
            lw = min(max(w.learned_score_weight, 0.0), 1.0)
            final = (1.0 - lw) * final + lw * learned_score
        return {
            "rel_score": float(rel_score),
            "struct_score": float(struct_score),
            "emb_score": float(emb_score),
            "rel_cond_emb_score": float(rel_cond_emb_score),
            "path_score": float(path_score),
            "type_score": float(type_score),
            "role_score": float(role_score),
            "hop_decay": float(hop_decay),
            "learned_score": float(learned_score),
            "learned_used": float(learned_used),
            "final_score": float(final),
        }

    def _relation_conditioned_embedding_score(self, head: int, query_rel: int, candidate: int,
                                              rel_score: float, fallback_emb: float) -> float:
        if self.emb_normed is None:
            return fallback_emb
        if not (head < len(self.emb_normed) and candidate < len(self.emb_normed)):
            return fallback_emb

        # If relation embeddings are available, use translational conditioning: sim(normalize(h + r), t).
        if self.rel_normed is not None and len(self.rel_normed) > 0:
            rel_base = query_rel % len(self.rel_normed)
            h_vec = self.emb_normed[head]
            r_vec = self.rel_normed[rel_base]
            query_vec = F.normalize(h_vec + r_vec, dim=-1)
            cand_vec = self.emb_normed[candidate]
            score = float(query_vec @ cand_vec)
            return max(0.0, score)

        # Fallback: relation-aware scaling when relation embeddings are unavailable.
        return float(fallback_emb * (0.5 + 0.5 * rel_score))

    def _path_support_score(self, query_rel: int, via_relations: List[int]) -> float:
        if not via_relations:
            return 0.0
        rel_sims = [self.kg.get_relation_similarity(query_rel, r) for r in via_relations]
        avg_rel = float(sum(rel_sims) / len(rel_sims))
        length_penalty = 1.0 / max(len(via_relations), 1)
        first_hop_bonus = rel_sims[0] if rel_sims else 0.0
        return 0.6 * avg_rel + 0.25 * first_hop_bonus + 0.15 * length_penalty

    def _type_consistency_score(self, query_rel: int, candidate: int) -> float:
        proto = self.relation_tail_prototypes.get(query_rel % self.kg.relation_num, [])
        if not proto:
            return 0.0
        cand_tokens = self._entity_tokens(candidate)
        if not cand_tokens:
            return 0.0
        proto_set = set(proto)
        overlap = len(cand_tokens & proto_set)
        return float(overlap / max(len(proto_set), 1))

    def _role_score(self, query_rel: int, candidate: int) -> float:
        neighbors = self.kg.get_neighbors(candidate)
        if not neighbors:
            return 0.0
        rel_hits = 0.0
        for _, r, _ in neighbors:
            rel_hits = max(rel_hits, self.kg.get_relation_similarity(query_rel, r))
        local_bridge = min(len(neighbors) / max(self._max_degree, 1), 1.0)
        return 0.6 * rel_hits + 0.4 * local_bridge

    def _redundancy_penalty(self, candidate: int, selected_nodes: List[int]) -> float:
        compare_nodes = selected_nodes[1:] if len(selected_nodes) > 1 else []
        if not self.config.enable_redundancy_penalty or not compare_nodes:
            return 0.0
        if self.emb_normed is not None and candidate < len(self.emb_normed):
            cand_vec = self.emb_normed[candidate]
            max_sim = 0.0
            for nid in compare_nodes:
                if nid == candidate or nid >= len(self.emb_normed):
                    continue
                max_sim = max(max_sim, float(cand_vec @ self.emb_normed[nid]))
            return self.config.redundancy_weight * max(0.0, max_sim)
        cand_tokens = self._entity_tokens(candidate)
        if not cand_tokens:
            return 0.0
        overlap_best = 0.0
        for nid in compare_nodes:
            toks = self._entity_tokens(nid)
            if not toks:
                continue
            overlap_best = max(overlap_best, len(cand_tokens & toks) / max(len(cand_tokens | toks), 1))
        return self.config.redundancy_weight * overlap_best

    def _entity_tokens(self, entity_id: int) -> set:
        if entity_id in self._token_cache:
            return self._token_cache[entity_id]
        text = self.entity_texts.get(entity_id, "")
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        self._token_cache[entity_id] = tokens
        return tokens

    def _compute_score_threshold(self, scored: List[Dict]) -> float:
        base = float(self.config.min_score_threshold)
        if not scored or not self.config.enable_dynamic_threshold:
            return base
        vals = [float(x["score"]) for x in scored]
        q = float(np.quantile(vals, self.config.dynamic_threshold_quantile))
        top = max(vals)
        top_ratio = float(top * self.config.dynamic_threshold_top_ratio)
        return max(base, q, top_ratio)

    def _build_hop_quota(self, budget: int) -> Dict[int, int]:
        if budget <= 0 or not self.config.use_hop_quota or self.config.max_hops <= 0:
            return {}

        quotas = {hop: 0 for hop in range(1, self.config.max_hops + 1)}
        if self.config.max_hops == 1:
            quotas[1] = budget
            return quotas

        q1 = int(round(budget * self.config.hop1_quota_ratio))
        q1 = max(1, min(q1, budget))
        quotas[1] = q1
        remain = budget - q1
        hops_rest = max(self.config.max_hops - 1, 1)
        base = remain // hops_rest
        for hop in range(2, self.config.max_hops + 1):
            quotas[hop] = base
        extra = remain - base * hops_rest
        hop = 2
        while extra > 0 and hop <= self.config.max_hops:
            quotas[hop] += 1
            extra -= 1
            hop += 1
        return quotas

    def _maybe_dump_debug(self, head: int, relation: int, scored: List[Dict],
                          selected_nodes: List[int], selected_meta: Dict,
                          hop_quota: Dict[int, int], score_threshold: float):
        if not self.config.selector_debug_log:
            return
        if self._debug_records >= self.config.selector_debug_max_records:
            return
        try:
            out_path = self.config.selector_debug_path
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            row = {
                "head": int(head),
                "relation": int(relation),
                "selected_nodes": [int(n) for n in selected_nodes],
                "selected_meta": {
                    str(k): {
                        "score": float(v["score"]),
                        "hop": int(v["hop"]),
                        "via": [int(x) for x in v["via"]],
                    } for k, v in selected_meta.items()
                },
                "hop_quota": {str(k): int(v) for k, v in hop_quota.items()},
                "score_threshold": float(score_threshold),
                "top_candidates": [
                    {
                        "node_id": int(x["node_id"]),
                        "hop": int(x["hop"]),
                        "score": float(x["score"]),
                        "via_relations": [int(v) for v in x["via_relations"]],
                        "score_detail": x["score_detail"],
                    } for x in scored[:30]
                ]
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._debug_records += 1
        except Exception:
            # Debug logging is best-effort and should never break training/data generation.
            return

    def _extract_key_paths(self, head: int, relation: int,
                           selected_nodes: List[int],
                           meta: Dict) -> List[List[Tuple[int, int, int]]]:
        paths = []
        node_set = set(selected_nodes)

        for h, r, t in self.kg.get_neighbors(head):
            if t in node_set:
                path = [(head, r, t)]
                paths.append(path)
                for h2, r2, t2 in self.kg.get_neighbors(t):
                    if t2 in node_set and t2 != head:
                        paths.append(path + [(t, r2, t2)])
        uniq = []
        seen = set()
        for p in paths:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                uniq.append(p)

        def _path_score(path):
            endpoint_score = sum(float(meta.get(step[-1], {}).get("score", 0.0)) for step in path) / max(len(path), 1)
            rel_match = 0.0
            for _, r, _ in path:
                rel_match = max(rel_match, self.kg.get_relation_similarity(relation, r))
            length_penalty = 1.0 / max(len(path), 1)
            return 0.5 * endpoint_score + 0.35 * rel_match + 0.15 * length_penalty

        uniq.sort(key=_path_score, reverse=True)
        paths = uniq
        return paths[:self.config.max_paths_in_prompt]
