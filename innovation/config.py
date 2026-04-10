"""
DSGR (Dynamic Subgraph Grounded Reasoning) 配置
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InnovationConfig:
    # --- Adaptive Subgraph Selection ---
    max_subgraph_nodes: int = 16
    max_hops: int = 2
    expand_top_per_hop: int = 48
    scorer_hidden_dim: int = 64
    relation_weight: float = 0.25
    structure_weight: float = 0.15
    embedding_weight: float = 0.15
    path_weight: float = 0.3
    type_weight: float = 0.2
    role_weight: float = 0.1
    redundancy_weight: float = 0.1
    min_score_threshold: float = 0.01
    enable_dynamic_threshold: bool = True
    dynamic_threshold_quantile: float = 0.7
    dynamic_threshold_top_ratio: float = 0.35
    use_hop_quota: bool = True
    hop1_quota_ratio: float = 0.6
    max_per_relation: int = 3
    learned_score_weight: float = 0.2
    relation_conditioned_embedding: bool = True
    enable_path_support: bool = True
    enable_type_consistency: bool = True
    enable_role_score: bool = True
    enable_redundancy_penalty: bool = True
    prototype_topk: int = 8
    selector_debug_log: bool = False
    selector_debug_path: str = "results_qwen/selector_debug.jsonl"
    selector_debug_max_records: int = 200

    # --- Dynamic Graph Token ---
    num_graph_tokens: int = 8
    token_dim: int = 128
    num_compress_heads: int = 4
    compress_dropout: float = 0.1

    # --- Structure-Aware CoT ---
    max_paths_in_prompt: int = 2
    max_evidence_items: int = 2
    include_reasoning_steps: bool = True
    cot_style: str = "graph_grounded"  # "graph_grounded" | "step_by_step" | "path_based"
    use_structured_prompt_blocks: bool = True
    include_graph_summary: bool = True
    include_candidate_hint: bool = False

    # --- Data ---
    data_path: str = "aligner/data/FB15k-237N"
    id2text_file: str = "id2text.txt"
    mid2id_file: str = "mid2id.txt"
    rel2id_file: str = "rel2id.txt"
