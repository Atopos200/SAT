"""
测试 M1 Pro 16GB 的极限：用原始论文参数跑 Stage 1，测速度和内存占用
"""
import os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.data import Data
from torch_geometric.utils import scatter

# 尝试 MPS 加速
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using: MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Using: CPU")

print(f"PyTorch: {torch.__version__}")

# ===== 导入 test_aligner_cpu 里的组件 =====
sys.path.insert(0, os.path.dirname(__file__))
from test_aligner_cpu import (
    GraphTransformer, CLIPModel, 
    get_id_map, get_id2text, load_triples, build_graph,
    TAGDataset, tokenize
)
from torch.utils.data import DataLoader

random.seed(42); np.random.seed(42); torch.manual_seed(42)

data_path = os.path.join("aligner", "data", "FB15k-237N")
ent2id = get_id_map(os.path.join(data_path, "mid2id.txt"))
rel2id = get_id_map(os.path.join(data_path, "rel2id.txt"))
id2text = get_id2text(os.path.join(data_path, "id2text.txt"))
entity_num, relation_num = len(ent2id), len(rel2id)

src_all, dst_all, rel_all, _ = load_triples(data_path, ['train', 'valid', 'test'], ent2id, rel2id)
graph = build_graph(src_all, dst_all, rel_all, entity_num, relation_num).to(DEVICE)

_, _, _, pos_tails_train = load_triples(data_path, ['train', 'valid'], ent2id, rel2id)

# ===== 测试不同配置 =====
configs = [
    {
        "name": "Dry-Run (小参数)",
        "embed_dim": 32, "t_width": 64, "t_layers": 2, "t_heads": 4,
        "gnn_dim": 32, "gt_layers": 2, "gt_heads": 4,
        "ctx_len": 64, "batch_size": 16, "n_queries": 500,
    },
    {
        "name": "中等参数",
        "embed_dim": 64, "t_width": 128, "t_layers": 4, "t_heads": 4,
        "gnn_dim": 64, "gt_layers": 2, "gt_heads": 4,
        "ctx_len": 64, "batch_size": 32, "n_queries": 2000,
    },
    {
        "name": "论文原始参数",
        "embed_dim": 128, "t_width": 512, "t_layers": 12, "t_heads": 8,
        "gnn_dim": 128, "gt_layers": 3, "gt_heads": 8,
        "ctx_len": 128, "batch_size": 64, "n_queries": 2000,
    },
]

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {cfg['name']}")
    print(f"  embed={cfg['embed_dim']}, t_width={cfg['t_width']}, t_layers={cfg['t_layers']}")
    print(f"  gnn={cfg['gnn_dim']}, gt_layers={cfg['gt_layers']}, batch={cfg['batch_size']}")

    try:
        model = CLIPModel(
            entity_num=entity_num, relation_num=relation_num,
            embed_dim=cfg['embed_dim'], context_length=cfg['ctx_len'],
            vocab_size=49408, t_width=cfg['t_width'], t_layers=cfg['t_layers'], t_heads=cfg['t_heads'],
            gnn_input=cfg['gnn_dim'], gnn_output=cfg['embed_dim'],
            gt_layers=cfg['gt_layers'], gt_head=cfg['gt_heads'],
            neigh_num=3, lr=1e-4
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        mem_mb = n_params * 4 / 1024 / 1024
        print(f"  Model params: {n_params:,} ({mem_mb:.1f} MB in fp32)")

        ds = TAGDataset(dict(list(pos_tails_train.items())[:cfg['n_queries']]), relation_num, 3)
        loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=TAGDataset.collate_fn)

        # 跑 5 个 step 测速
        model.train()
        step_times = []
        for step, (src, rel, dst) in enumerate(loader):
            if step >= 5:
                break
            t0 = time.time()
            src_arr, dst_arr = src.numpy(), dst.numpy().reshape(-1)
            src_texts = tokenize([id2text.get(i, "") for i in src_arr], context_length=cfg['ctx_len']).to(DEVICE)
            dst_texts = tokenize([id2text.get(j, "") for j in dst_arr], context_length=cfg['ctx_len']).to(DEVICE)
            src, rel, dst = src.to(DEVICE), rel.to(DEVICE), dst.to(DEVICE)

            s_g, s_t, t_t, labels = model(graph, src, rel, dst, src_texts, dst_texts, DEVICE)
            loss = model.align_loss(s_g, s_t, labels) + model.align_loss(s_g, t_t, labels)
            model.optim.zero_grad()
            loss.backward()
            model.optim.step()
            if DEVICE.type == 'mps':
                torch.mps.synchronize()
            
            elapsed = time.time() - t0
            step_times.append(elapsed)

        avg_step = np.mean(step_times)
        total_queries = len(pos_tails_train) * 2
        steps_per_epoch = total_queries // cfg['batch_size']
        
        print(f"  Avg step time: {avg_step:.3f} s")
        print(f"  Steps/epoch (full data): {steps_per_epoch}")
        epoch_time_min = steps_per_epoch * avg_step / 60
        print(f"  Est. 1 epoch (full data): {epoch_time_min:.1f} min")
        print(f"  Est. 100 epochs: {epoch_time_min * 100 / 60:.1f} hours ({epoch_time_min * 100 / 60 / 24:.1f} days)")
        print(f"  Status: OK")

    except Exception as e:
        print(f"  FAILED: {e}")

    del model
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    import gc; gc.collect()

# ===== LLM 可行性评估 =====
print(f"\n{'='*60}")
print("Stage 2: LLM 可行性评估 (16GB 内存)")
print("="*60)

llm_options = [
    ("Llama-2-7B (论文原版)", 7000, "不可行"),
    ("Llama-2-7B 4-bit量化", 3500, "勉强"),
    ("Qwen2.5-3B", 3000, "可行"),
    ("TinyLlama-1.1B", 1100, "推荐"),
    ("Qwen2.5-1.5B", 1500, "推荐"),
    ("Qwen2.5-0.5B", 500, "轻松"),
]

print(f"{'模型':<30} {'参数量':>10} {'FP16显存':>10} {'4-bit显存':>10} {'可行性':>8}")
print("-" * 75)
for name, params_m, status in llm_options:
    fp16_gb = params_m * 2 / 1024
    int4_gb = params_m * 0.5 / 1024
    print(f"{name:<30} {params_m:>7}M {fp16_gb:>8.1f}GB {int4_gb:>8.1f}GB {status:>8}")

print(f"\n你的可用内存: ~12GB (16GB 减去系统占用)")
print("结论: Stage 2 推荐使用 TinyLlama-1.1B 或 Qwen2.5-1.5B 替代 Llama-2-7B")
