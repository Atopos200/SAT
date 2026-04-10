"""
SAT Stage 2: Predictor Dry-Run on CPU
用极小随机 Llama 模型在 CPU 上跑通 Predictor 全流程。
必须先运行 test_aligner_cpu.py 生成 checkpoints_cpu/ 产出物。
用法: conda activate sat_cpu && cd SAT && python test_predictor_cpu.py
"""
import os
import sys
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from tqdm import tqdm

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from transformers import TrainingArguments, Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

IGNORE_INDEX = -100
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


# ==================== 1. Graph Transformer (与 Stage 1 一致) ====================
class GTLayer(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.qTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
        self.kTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
        self.vTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, g, embeds):
        rows, cols = g.edge_index
        nvar = embeds.shape[0]
        rowE, colE = embeds[rows], embeds[cols]
        evar = rowE.shape[0]
        hd = self.d_model // self.n_head
        qE = (rowE @ self.qTrans).view(evar, self.n_head, hd)
        kE = (colE @ self.kTrans).view(evar, self.n_head, hd)
        vE = (colE @ self.vTrans).view(evar, self.n_head, hd)
        att = torch.clamp(torch.einsum("ehd,ehd->eh", qE, kE), -10, 10)
        expAtt = torch.exp(att)
        attNorm = scatter(expAtt, rows, dim=0, dim_size=nvar, reduce='sum')[rows]
        att = expAtt / (attNorm + 1e-8)
        resE = torch.einsum("eh,ehd->ehd", att, vE).view(evar, self.d_model)
        resE = scatter(resE, rows, dim=0, dim_size=nvar, reduce='sum')
        return self.norm(resE + embeds)


class GraphEncoder(nn.Module):
    def __init__(self, entity_num, relation_num, d_in, d_out, n_layers, n_head):
        super().__init__()
        self.entity_embedding = nn.Embedding(entity_num, d_in)
        self.relation_embedding = nn.Embedding(relation_num * 2, d_in)
        self.input_fc = nn.Linear(d_in, d_in)
        self.gt_layers = nn.ModuleList([GTLayer(d_in, n_head) for _ in range(n_layers)])
        self.output_fc = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, graph):
        x = self.entity_embedding(graph.entity)
        z = self.input_fc(x)
        embeds = self.dropout(z)
        for gt in self.gt_layers:
            embeds = gt(graph, embeds)
        return self.output_fc(embeds)


# ==================== 2. GraphLlama (简化版) ====================
class GraphLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, graph_encoder, graph_hidden_size):
        super().__init__(config)
        self.graph_encoder = graph_encoder
        self.graph_projector = nn.Linear(graph_hidden_size, config.hidden_size)
        self.graph_patch_token_id = None
        self.graph_start_token_id = None
        self.graph_end_token_id = None

    def encode_graph(self, graph_data):
        with torch.no_grad():
            return self.graph_encoder(graph_data)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                graph_data=None, **kwargs):
        inputs_embeds = self.model.embed_tokens(input_ids)

        if graph_data is not None and self.graph_start_token_id is not None:
            for batch_idx in range(input_ids.shape[0]):
                cur_ids = input_ids[batch_idx]
                if self.graph_start_token_id not in cur_ids:
                    continue
                g = graph_data[batch_idx]
                node_feats = self.encode_graph(g)
                node_feats = self.graph_projector(node_feats)
                start_pos = (cur_ids == self.graph_start_token_id).nonzero(as_tuple=True)[0]
                if len(start_pos) == 0:
                    continue
                sp = start_pos[0].item()
                n_nodes = node_feats.shape[0]
                inputs_embeds[batch_idx, sp + 1: sp + 1 + n_nodes] = node_feats.to(inputs_embeds.dtype)

        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size),
                                   shift_labels.view(-1), ignore_index=IGNORE_INDEX)
        return {"loss": loss, "logits": logits}


# ==================== 3. Dataset ====================
class GraphInstructDataset(Dataset):
    def __init__(self, data_path, graph_data_all, tokenizer, max_len, max_samples=200):
        with open(data_path) as f:
            self.data = json.load(f)[:max_samples]
        self.graph_data_all = graph_data_all
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        graph_type = item['id'].split('_')[0]
        graph_dict = item['graph']
        node_list = graph_dict['node_list']
        edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
        target_node = graph_dict['node_idx']
        node_rep = self.graph_data_all[graph_type].x[node_list]
        n_nodes = len(node_list)

        question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']

        replace_token = DEFAULT_G_START_TOKEN + DEFAULT_GRAPH_PATCH_TOKEN * n_nodes + DEFAULT_G_END_TOKEN
        question = question.replace(DEFAULT_GRAPH_TOKEN, replace_token)

        prompt = f"USER: {question} ASSISTANT: {answer}</s>"
        encoded = self.tokenizer(prompt, max_length=self.max_len, truncation=True,
                                 padding="max_length", return_tensors="pt")
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)

        sep = "ASSISTANT: "
        sep_pos = prompt.find(sep)
        if sep_pos >= 0:
            prefix = prompt[:sep_pos + len(sep)]
            prefix_len = len(self.tokenizer(prefix, truncation=True).input_ids)
        else:
            prefix_len = len(input_ids) // 2

        labels = input_ids.clone()
        labels[:prefix_len] = IGNORE_INDEX
        labels[attention_mask == 0] = IGNORE_INDEX

        graph = Data(graph_node=node_rep, edge_index=edge_index, 
                     entity=torch.arange(n_nodes), target_node=torch.tensor([target_node]))

        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "graph_data": graph}


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    graph_data = [b["graph_data"] for b in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "graph_data": graph_data}


# ==================== 4. Main ====================
def main():
    ckpt_dir = os.path.join("checkpoints_cpu", "FB15k-237N")
    assert os.path.exists(ckpt_dir), f"请先运行 test_aligner_cpu.py 生成 {ckpt_dir}"

    with open(os.path.join(ckpt_dir, "config.json")) as f:
        aligner_cfg = json.load(f)

    GRAPH_DIM = aligner_cfg["gnn_output"]
    LLM_HIDDEN = 256
    LLM_LAYERS = 2
    LLM_HEADS = 4
    LLM_INTERMEDIATE = 512
    VOCAB_SIZE = 32000
    MAX_LEN = 256
    BATCH_SIZE = 2
    EPOCHS = 2
    LR = 1e-4
    MAX_SAMPLES = 100

    device = torch.device("cpu")

    # --- 1. 加载图数据 ---
    logging.info("Loading graph data from Stage 1...")
    graph_data_all = torch.load(os.path.join(ckpt_dir, "graph_data_all.pt"), weights_only=False)
    logging.info(f"  Graph keys: {list(graph_data_all.keys())}")

    # --- 2. 创建图编码器并加载权重 ---
    logging.info("Building graph encoder...")
    graph_encoder = GraphEncoder(
        entity_num=aligner_cfg["entity_num"],
        relation_num=aligner_cfg["relation_num"],
        d_in=aligner_cfg["gnn_input"],
        d_out=GRAPH_DIM,
        n_layers=aligner_cfg["gt_layers"],
        n_head=aligner_cfg["gt_head"]
    )
    aligner_state = torch.load(os.path.join(ckpt_dir, "aligner_best.pkl"),
                               map_location="cpu", weights_only=False)
    gnn_state = {k.replace("gnn.", ""): v for k, v in aligner_state.items() if k.startswith("gnn.")}
    graph_encoder.load_state_dict(gnn_state)
    graph_encoder.eval()
    graph_encoder.requires_grad_(False)
    logging.info("  Graph encoder loaded and frozen.")

    # --- 3. 创建极小 Llama (随机初始化，只验证流程) ---
    logging.info("Building tiny Llama model (random init, for pipeline test only)...")
    llama_config = LlamaConfig(
        vocab_size=VOCAB_SIZE + 10,
        hidden_size=LLM_HIDDEN,
        intermediate_size=LLM_INTERMEDIATE,
        num_hidden_layers=LLM_LAYERS,
        num_attention_heads=LLM_HEADS,
        max_position_embeddings=MAX_LEN + 128,
    )
    model = GraphLlamaForCausalLM(llama_config, graph_encoder, GRAPH_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"  Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # --- 4. 创建 Tokenizer (用 Llama 格式，但随机 vocab) ---
    logging.info("Creating tokenizer...")
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    tok_model = models.BPE()
    tokenizer_obj = Tokenizer(tok_model)
    tokenizer_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer_obj = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>",
                        DEFAULT_GRAPH_PATCH_TOKEN, DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN]
    )

    sample_texts = []
    data_path = os.path.join("predictor", "data_llm_lp", "FB15k-237N", "train.json")
    with open(data_path) as f:
        raw = json.load(f)[:500]
    for item in raw:
        for conv in item.get("conversations", []):
            sample_texts.append(conv["value"])
    tokenizer_obj.train_from_iterator(sample_texts, trainer=trainer_obj)

    tok_save_dir = os.path.join(ckpt_dir, "tokenizer_cpu")
    os.makedirs(tok_save_dir, exist_ok=True)
    tokenizer_obj.save(os.path.join(tok_save_dir, "tokenizer.json"))

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tok_save_dir, "tokenizer.json"),
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [
        DEFAULT_GRAPH_PATCH_TOKEN, DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN
    ]})
    model.resize_token_embeddings(len(tokenizer))

    patch_id = tokenizer.convert_tokens_to_ids(DEFAULT_GRAPH_PATCH_TOKEN)
    start_id = tokenizer.convert_tokens_to_ids(DEFAULT_G_START_TOKEN)
    end_id = tokenizer.convert_tokens_to_ids(DEFAULT_G_END_TOKEN)
    model.graph_patch_token_id = patch_id
    model.graph_start_token_id = start_id
    model.graph_end_token_id = end_id
    logging.info(f"  Vocab size: {len(tokenizer)}, patch={patch_id}, start={start_id}, end={end_id}")

    # --- 5. 加载指令数据 ---
    logging.info("Loading instruction data...")
    train_ds = GraphInstructDataset(data_path, graph_data_all, tokenizer, MAX_LEN, MAX_SAMPLES)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    logging.info(f"  Train samples: {len(train_ds)}")

    # --- 6. 训练 (只训 graph_projector) ---
    logging.info("=" * 50)
    logging.info("Stage 2: Predictor Training (CPU Dry-Run)")
    logging.info("  Only training: graph_projector")
    logging.info("=" * 50)

    model.requires_grad_(False)
    for p in model.graph_projector.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.graph_projector.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"  Epoch {epoch + 1}")):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                graph_data=batch["graph_data"]
            )
            loss = out["loss"]
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 10 == 0:
                logging.info(f"    step {step}, loss={loss.item():.4f}")
        avg_loss = total_loss / max(len(train_loader), 1)
        logging.info(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

    # --- 7. 测试推理 ---
    logging.info("\nTesting inference...")
    model.eval()
    test_item = train_ds[0]
    with torch.no_grad():
        input_ids = test_item["input_ids"].unsqueeze(0)
        attention_mask = test_item["attention_mask"].unsqueeze(0)
        graph = test_item["graph_data"]

        out = model(input_ids=input_ids, attention_mask=attention_mask, graph_data=[graph])
        logits = out["logits"]
        pred_ids = torch.argmax(logits[0, -20:], dim=-1)
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        logging.info(f"  Sample prediction (last 20 tokens): {pred_text[:200]}")

    # --- 8. 保存 ---
    save_path = os.path.join(ckpt_dir, "predictor_projector.pt")
    torch.save(model.graph_projector.state_dict(), save_path)

    logging.info(f"\n{'=' * 50}")
    logging.info("Stage 2 DONE!")
    logging.info(f"  Projector saved to: {save_path}")
    logging.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
