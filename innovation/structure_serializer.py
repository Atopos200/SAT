"""
Structure-Aware Serializer
将选中的子图同时转换为：
  1. 动态 graph tokens（给图编码器 + LLM）
  2. 结构感知 CoT 推理提示（给 LLM 文本输入）

两个输出共享同一个 selected subgraph，确保语义一致。
"""
import torch
from typing import Dict, List, Tuple, Optional

from innovation.config import InnovationConfig
from innovation.subgraph_selector import SelectedSubgraph


class StructureAwareSerializer:
    """
    将 SelectedSubgraph 序列化为 LLM 可消费的格式。

    负责两件事：
      1. 生成结构感知的 CoT prompt
      2. 准备 graph token 所需的节点排序和截取
    """

    def __init__(self, config: InnovationConfig,
                 id2entity_name: Dict[int, str],
                 id2relation_name: Dict[int, str]):
        self.config = config
        self.ent_names = id2entity_name
        self.rel_names = id2relation_name

    def get_entity_name(self, eid: int) -> str:
        name = self.ent_names.get(eid, f"Entity_{eid}")
        if len(name) > 80:
            name = name[:77] + "..."
        return name

    def get_relation_name(self, rid: int) -> str:
        return self.rel_names.get(rid % len(self.rel_names) if self.rel_names else rid,
                                  f"Relation_{rid}")

    def serialize(self, subgraph: SelectedSubgraph, head: int, relation: int,
                  ) -> Dict:
        """
        核心方法：将子图转换为 LLM 输入。

        Returns:
            {
                "cot_prompt": str,          # 结构感知推理提示
                "node_order": List[int],    # 按重要性排序的节点 ID
                "edge_index": Tensor,       # 子图边索引
                "importance": Tensor,       # 节点重要性分数
                "graph_context": str,       # 简洁的图上下文描述
            }
        """
        structured_prompt = self._build_structured_prompt(subgraph, head, relation)
        cot_prompt = self._build_cot_prompt(subgraph, head, relation)
        graph_context = self._build_graph_context(subgraph, head, relation)
        graph_summary = self._build_graph_summary(subgraph, head, relation)

        sorted_indices = torch.argsort(subgraph.importance_scores, descending=True)
        node_order = [subgraph.node_ids[i] for i in sorted_indices.tolist()]

        return {
            "cot_prompt": cot_prompt,
            "structured_prompt": structured_prompt,
            "node_order": node_order,
            "edge_index": subgraph.edge_index,
            "importance": subgraph.importance_scores,
            "graph_context": graph_context,
            "graph_summary": graph_summary,
        }

    def _build_structured_prompt(self, sg: SelectedSubgraph, head: int, relation: int) -> str:
        head_name = self.get_entity_name(head)
        rel_name = self.get_relation_name(relation)
        lines = []
        lines.append("[QUERY]")
        lines.append(f"({head_name}, {rel_name}, ?)")
        lines.append("")
        lines.append("[LOCAL_SUBGRAPH]")
        lines.append(f"nodes = {len(sg.node_ids)}")
        lines.append(f"edges = {sg.edge_index.shape[1]}")

        direct_triples = []
        for path in sg.paths:
            if path and path[0][0] == head:
                h, r, t = path[0]
                direct_triples.append(
                    f"({self.get_entity_name(h)}, {self.get_relation_name(r)}, {self.get_entity_name(t)})"
                )
        if direct_triples:
            lines.append("")
            lines.append("[TOP_NEIGHBORS]")
            for row in direct_triples[:self.config.max_evidence_items]:
                lines.append(row)

        multi_hop = [p for p in sg.paths if len(p) > 1]
        if multi_hop:
            lines.append("")
            lines.append("[TOP_PATHS]")
            for path in multi_hop[:self.config.max_paths_in_prompt]:
                parts = []
                for h, r, t in path:
                    parts.append(f"({self.get_entity_name(h)}, {self.get_relation_name(r)}, {self.get_entity_name(t)})")
                lines.append(" -> ".join(parts))

        top_entities = []
        sorted_idx = torch.argsort(sg.importance_scores, descending=True)
        for idx in sorted_idx.tolist():
            nid = sg.node_ids[idx]
            if nid != head:
                top_entities.append(self.get_entity_name(nid))
        if top_entities:
            lines.append("")
            lines.append("[KEY_ENTITIES]")
            lines.append(", ".join(top_entities[:self.config.max_evidence_items + 1]))

        lines.append("")
        lines.append("[TASK]")
        lines.append("Select the most plausible tail entity for the query.")
        return "\n".join(lines)

    def _build_cot_prompt(self, subgraph: SelectedSubgraph, head: int, relation: int) -> str:
        head_name = self.get_entity_name(head)
        rel_name = self.get_relation_name(relation)

        if self.config.cot_style == "graph_grounded":
            return self._cot_graph_grounded(subgraph, head, head_name, rel_name)
        elif self.config.cot_style == "path_based":
            return self._cot_path_based(subgraph, head, head_name, rel_name)
        else:
            return self._cot_step_by_step(subgraph, head, head_name, rel_name)

    def _cot_graph_grounded(self, sg: SelectedSubgraph, head: int,
                            head_name: str, rel_name: str) -> str:
        lines = []
        lines.append(f"Query: ({head_name}, {rel_name}, ?)")
        lines.append("Evidence:")
        direct_connections = []
        for path in sg.paths:
            if path and path[0][0] == head:
                h, r, t = path[0]
                t_name = self.get_entity_name(t)
                r_name = self.get_relation_name(r)
                score = 0.0
                for i, nid in enumerate(sg.node_ids):
                    if nid == t:
                        score = sg.importance_scores[i].item()
                        break
                direct_connections.append((r_name, t_name, score))

        if direct_connections:
            direct_connections.sort(key=lambda x: x[2], reverse=True)
            for rn, tn, sc in direct_connections[:self.config.max_evidence_items]:
                lines.append(f"- {head_name} --[{rn}]--> {tn}")
        multi_hop = [p for p in sg.paths if len(p) > 1]
        if multi_hop:
            for path in multi_hop[:self.config.max_paths_in_prompt]:
                path_str_parts = []
                for h, r, t in path:
                    hn = self.get_entity_name(h)
                    rn = self.get_relation_name(r)
                    tn = self.get_entity_name(t)
                    path_str_parts.append(f"{hn} --[{rn}]--> {tn}")
                lines.append(f"- Path: {' -> '.join(path_str_parts)}")

        top_neighbors = []
        sorted_idx = torch.argsort(sg.importance_scores, descending=True)
        for i in sorted_idx[:self.config.max_evidence_items + 1].tolist():
            nid = sg.node_ids[i]
            if nid != head:
                top_neighbors.append(self.get_entity_name(nid))
        if top_neighbors:
            lines.append(f"- Key entities: {', '.join(top_neighbors[:self.config.max_evidence_items])}")
        lines.append("Answer with only the tail entity name.")

        return "\n".join(lines)

    def _cot_path_based(self, sg: SelectedSubgraph, head: int,
                        head_name: str, rel_name: str) -> str:
        lines = []
        lines.append(f"Question: ({head_name}, {rel_name}, ?)")
        lines.append("")
        lines.append("Reasoning over knowledge graph paths:")
        lines.append("")

        for i, path in enumerate(sg.paths[:self.config.max_paths_in_prompt]):
            path_parts = []
            for h, r, t in path:
                hn = self.get_entity_name(h)
                rn = self.get_relation_name(r)
                tn = self.get_entity_name(t)
                path_parts.append(f"({hn}) --[{rn}]--> ({tn})")
            lines.append(f"  Path {i + 1}: {' , '.join(path_parts)}")

        lines.append("")
        lines.append("Based on the above graph paths, the answer is:")
        return "\n".join(lines)

    def _cot_step_by_step(self, sg: SelectedSubgraph, head: int,
                          head_name: str, rel_name: str) -> str:
        lines = []
        lines.append(f"Question: ({head_name}, {rel_name}, ?)")
        lines.append(f"Subgraph nodes: {len(sg.node_ids)}, Edges: {sg.edge_index.shape[1]}")
        lines.append("")
        lines.append("Let me analyze the graph structure step by step.")
        lines.append(f"1. The anchor entity is {head_name}.")
        lines.append(f"2. We need to find what connects to it via {rel_name}.")

        top_idx = torch.argsort(sg.importance_scores, descending=True)
        top_names = [self.get_entity_name(sg.node_ids[i]) for i in top_idx[:4].tolist()
                     if sg.node_ids[i] != head]
        if top_names:
            lines.append(f"3. The most structurally relevant neighbors are: {', '.join(top_names)}.")

        lines.append("4. Based on this analysis, the answer is:")
        return "\n".join(lines)

    def _build_graph_context(self, sg: SelectedSubgraph, head: int, relation: int) -> str:
        head_name = self.get_entity_name(head)
        rel_name = self.get_relation_name(relation)

        parts = [f"Query: ({head_name}, {rel_name}, ?)"]
        parts.append(f"Subgraph: {len(sg.node_ids)} nodes, {sg.edge_index.shape[1]} edges")

        top_idx = torch.argsort(sg.importance_scores, descending=True)
        top_names = []
        for i in top_idx[:5].tolist():
            nid = sg.node_ids[i]
            if nid != head:
                top_names.append(self.get_entity_name(nid))
        if top_names:
            parts.append(f"Key entities: {', '.join(top_names)}")

        return " | ".join(parts)

    def _build_graph_summary(self, sg: SelectedSubgraph, head: int, relation: int) -> str:
        head_name = self.get_entity_name(head)
        rel_name = self.get_relation_name(relation)
        sorted_idx = torch.argsort(sg.importance_scores, descending=True)
        key_entities = []
        for idx in sorted_idx.tolist():
            nid = sg.node_ids[idx]
            if nid != head:
                key_entities.append(self.get_entity_name(nid))
        key_entities = key_entities[:3]

        path_summaries = []
        for path in sg.paths[:self.config.max_paths_in_prompt]:
            rel_chain = [self.get_relation_name(r) for _, r, _ in path]
            path_summaries.append(" > ".join(rel_chain))
        path_text = "; ".join(path_summaries) if path_summaries else "none"
        entity_text = ", ".join(key_entities) if key_entities else "none"
        return (
            f"query=({head_name}, {rel_name}, ?); "
            f"nodes={len(sg.node_ids)}; edges={sg.edge_index.shape[1]}; "
            f"key_entities={entity_text}; path_patterns={path_text}"
        )


def build_cot_instruction(
    cot_prompt: str,
    answer: str,
    graph_token_placeholder: str = "<graph>",
    graph_summary: str = "",
) -> Dict:
    """
    构造一条完整的 CoT 指令数据（用于训练）。

    将 cot_prompt 和 graph token 占位符融合为一条对话：
      human: [graph context] + [CoT reasoning prompt]
      gpt:   [reasoning steps] + [answer]
    """
    prefix = f"Graph tokens: {graph_token_placeholder}\n"
    if graph_summary:
        prefix += f"[GRAPH_SUMMARY]\n{graph_summary}\n\n"
    question = prefix + f"{cot_prompt}\nOutput only the final entity name."

    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]
    }
