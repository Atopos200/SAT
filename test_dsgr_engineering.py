import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict

from innovation.config import InnovationConfig
from innovation.subgraph_selector import AdaptiveSubgraphSelector, KGIndex
from innovation import build_cot_data
from run_full_qwen import build_filtered_candidate_ids


class TestFilteredRankingProtocol(unittest.TestCase):
    def test_filtered_candidates_remove_other_true_tails(self):
        item = {"graph": {"node_list": [1, 2, 3, 4]}}
        true_tails_by_hr = {(10, 5): {1, 2, 3}}
        tails_by_rel = {5: {1, 2, 3, 4, 7}}
        all_entity_ids = [1, 2, 3, 4, 5, 6, 7]

        out = build_filtered_candidate_ids(
            item=item,
            h_id=10,
            r_id=5,
            t_id=2,
            true_tails_by_hr=true_tails_by_hr,
            tails_by_rel=tails_by_rel,
            all_entity_ids=all_entity_ids,
            mode="relation",
            max_candidates=0,
        )
        self.assertIn(2, out)   # true tail must be kept
        self.assertNotIn(1, out)  # other true tails are filtered
        self.assertNotIn(3, out)
        self.assertIn(4, out)

    def test_filtered_candidates_max_candidates_still_keep_truth(self):
        item = {"graph": {"node_list": [0, 1, 2]}}
        true_tails_by_hr = {(1, 1): {9}}
        tails_by_rel = {1: set(range(50))}
        all_entity_ids = list(range(50))
        out = build_filtered_candidate_ids(
            item=item,
            h_id=1,
            r_id=1,
            t_id=9,
            true_tails_by_hr=true_tails_by_hr,
            tails_by_rel=tails_by_rel,
            all_entity_ids=all_entity_ids,
            mode="relation",
            max_candidates=5,
        )
        self.assertIn(9, out)
        self.assertLessEqual(len(out), 5)


class TestBuildCotStats(unittest.TestCase):
    class _FakeSelector:
        def select(self, head, relation, max_nodes=None):
            if head == 999:
                raise ValueError("synthetic failure")
            class SG:
                edge_index = __import__("torch").tensor([[0], [0]])
                node_ids = [head]
                importance_scores = __import__("torch").tensor([1.0])
                paths = []
            return SG()

    class _FakeSerializer:
        def serialize(self, subgraph, head, relation):
            return {"cot_prompt": f"Q({head},{relation})"}

    def test_process_from_instruction_stats(self):
        cfg = InnovationConfig()
        cfg._max_samples = -1
        selector = self._FakeSelector()
        serializer = self._FakeSerializer()
        data = [
            {"id": "a_1_x", "graph": {"node_list": [0], "node_idx": 0, "edge_index": [[0], [0]]},
             "conversations": [{"value": "q"}, {"value": "ans"}]},
            {"id": "b_2_x", "conversations": [{"value": "q"}, {"value": "ans"}]},  # no graph -> fallback
            {"id": "c_3_x", "graph": {"node_list": [999], "node_idx": 0, "edge_index": [[0], [0]]},
             "conversations": [{"value": "q"}, {"value": "ans"}]},  # failure -> fallback + error
        ]
        out, stats = build_cot_data._process_from_instruction("train", data, cfg, selector, serializer)
        self.assertEqual(len(out), 3)
        self.assertEqual(stats["total_input"], 3)
        self.assertEqual(stats["total_output"], 3)
        self.assertEqual(stats["success"], 1)
        self.assertEqual(stats["fallback"], 2)
        self.assertIn("ValueError", stats["error_types"])


class TestSelectorEngineering(unittest.TestCase):
    def test_selector_debug_and_relation_quota(self):
        triples = [
            (0, 0, 1), (0, 0, 2), (0, 1, 3),  # one-hop
            (1, 2, 4), (2, 2, 5), (3, 2, 6),  # two-hop
        ]
        with tempfile.TemporaryDirectory() as td:
            debug_path = os.path.join(td, "selector_debug.jsonl")
            cfg = InnovationConfig(
                max_subgraph_nodes=5,
                max_hops=2,
                use_hop_quota=True,
                hop1_quota_ratio=0.5,
                max_per_relation=1,
                selector_debug_log=True,
                selector_debug_path=debug_path,
                selector_debug_max_records=5,
            )
            kg = KGIndex(triples, entity_num=7, relation_num=3)
            selector = AdaptiveSubgraphSelector(kg, cfg, entity_embeddings=None)
            sg = selector.select(head=0, relation=0, max_nodes=5)
            self.assertGreaterEqual(len(sg.node_ids), 1)
            self.assertTrue(os.path.exists(debug_path))

            with open(debug_path, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())
            self.assertIn("top_candidates", row)
            self.assertIn("score_detail", row["top_candidates"][0])
            self.assertIn("hop_quota", row)

            rel_counts = defaultdict(int)
            for nid, meta in row["selected_meta"].items():
                if int(meta["hop"]) == 0:
                    continue
                via = meta.get("via", [])
                if via:
                    rel_counts[via[0]] += 1
            self.assertTrue(all(v <= 1 for v in rel_counts.values()))

    def test_dynamic_threshold_and_relation_conditioned_score(self):
        triples = [(0, 0, 1), (0, 1, 2), (1, 2, 3), (2, 2, 4)]
        import torch
        ent_emb = torch.randn(5, 8)
        rel_emb = torch.randn(3, 8)
        cfg = InnovationConfig(
            max_subgraph_nodes=4,
            max_hops=2,
            enable_dynamic_threshold=True,
            dynamic_threshold_quantile=0.6,
            dynamic_threshold_top_ratio=0.3,
            relation_conditioned_embedding=True,
        )
        kg = KGIndex(triples, entity_num=5, relation_num=3)
        selector = AdaptiveSubgraphSelector(kg, cfg, entity_embeddings=ent_emb, relation_embeddings=rel_emb)
        scored = [
            {"score": 0.10},
            {"score": 0.15},
            {"score": 0.50},
            {"score": 0.30},
        ]
        th = selector._compute_score_threshold(scored)
        self.assertGreaterEqual(th, cfg.min_score_threshold)
        self.assertGreater(th, 0.10)

        detail = selector._score_candidate(head=0, query_rel=1, candidate=2, hop=1, via_relations=[1])
        self.assertIn("rel_cond_emb_score", detail)
        self.assertIn("final_score", detail)

    def test_learned_scorer_is_wired(self):
        triples = [(0, 0, 1), (0, 1, 2), (1, 2, 3)]
        import torch
        ent_emb = torch.randn(4, 8)
        rel_emb = torch.randn(3, 8)
        cfg = InnovationConfig(
            max_subgraph_nodes=4,
            max_hops=2,
            learned_score_weight=0.4,
        )
        kg = KGIndex(triples, entity_num=4, relation_num=3)
        selector = AdaptiveSubgraphSelector(
            kg, cfg, entity_embeddings=ent_emb, relation_embeddings=rel_emb, mode="learned"
        )
        detail = selector._score_candidate(head=0, query_rel=1, candidate=2, hop=1, via_relations=[1])
        self.assertEqual(detail["learned_used"], 1.0)
        self.assertGreaterEqual(detail["learned_score"], 0.0)
        self.assertLessEqual(detail["learned_score"], 1.0)


class TestManifestGeneration(unittest.TestCase):
    def test_manifest_fields_exist(self):
        data_path = os.path.join("aligner", "data", "FB15k-237N")
        pred_path = os.path.join("predictor", "data_llm_lp", "FB15k-237N")
        if not (os.path.exists(data_path) and os.path.exists(pred_path)):
            self.skipTest("Required dataset paths not found.")

        with tempfile.TemporaryDirectory() as td:
            cmd = [
                sys.executable, "-m", "innovation.build_cot_data",
                "--output_dir", td,
                "--max_samples", "1",
            ]
            subprocess.check_call(cmd, cwd=os.path.dirname(__file__))
            manifest_path = os.path.join(td, "manifest.json")
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertIn("created_at", manifest)
            self.assertIn("args", manifest)
            self.assertIn("config", manifest)
            self.assertIn("kg", manifest)
            self.assertIn("splits", manifest)
            for split in ["train", "valid", "test"]:
                self.assertIn(split, manifest["splits"])
                s = manifest["splits"][split]
                for k in ["total_input", "total_output", "success", "fallback", "skipped", "error_types"]:
                    self.assertIn(k, s)


if __name__ == "__main__":
    unittest.main()
