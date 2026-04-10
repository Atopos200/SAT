"""
SAT Stage 1: Aligner Dry-Run on CPU
用小参数在 CPU 上跑通 Aligner 全流程，验证代码逻辑正确。
用法: conda activate sat_cpu && cd SAT && python test_aligner_cpu.py
"""
import os
import sys
import random
import math
import logging
import gzip
import html
import json
import hashlib
import regex as re
from functools import lru_cache
from collections import OrderedDict, defaultdict
from itertools import chain
from random import choice

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from sklearn.metrics import accuracy_score


# ==================== 1. SimpleTokenizer (内联，避免路径问题) ====================
@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer:
    def __init__(self, bpe_path=None):
        if bpe_path is None:
            bpe_path = os.path.join(os.path.dirname(__file__), "aligner", "model", "bpe_simple_vocab_16e6.txt.gz")
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        self.byte_encoder = bytes_to_unicode()

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)
        if not pairs:
            return token + '</w>'
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens


_tokenizer = SimpleTokenizer()


def tokenize(texts, context_length=64, truncate=True):
    if isinstance(texts, str):
        texts = [texts]
    sot = _tokenizer.encoder["<|startoftext|>"]
    eot = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot] + _tokenizer.encode(text) + [eot] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


# ==================== 2. Graph Transformer ====================
def Mv2Same(vars_list):
    return [v.to(vars_list[0].device) for v in vars_list]


class GTLayer(nn.Module):
    def __init__(self, d_model, n_head, att_norm=True):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.qTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
        self.kTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
        self.vTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
        if att_norm:
            self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.att_norm = att_norm

    def forward(self, g, embeds):
        rows, cols = g.edge_index
        nvar, _ = embeds.shape
        rowE = embeds[rows]
        colE = embeds[cols]
        evar = rowE.shape[0]
        hd = self.d_model // self.n_head

        qE = (rowE @ self.qTrans).view(evar, self.n_head, hd)
        kE = (colE @ self.kTrans).view(evar, self.n_head, hd)
        vE = (colE @ self.vTrans).view(evar, self.n_head, hd)

        att = torch.einsum("ehd, ehd -> eh", qE, kE)
        att = torch.clamp(att, -10.0, 10.0)
        expAtt = torch.exp(att)
        attNorm = scatter(expAtt, rows, dim=0, dim_size=nvar, reduce='sum')[rows]
        att = expAtt / (attNorm + 1e-8)

        resE = torch.einsum("eh, ehd -> ehd", att, vE).view(evar, self.d_model)
        resE = scatter(resE, rows, dim=0, dim_size=nvar, reduce='sum')
        resE = resE + embeds
        if self.att_norm:
            resE = self.norm(resE)
        return resE


class GraphTransformer(nn.Module):
    def __init__(self, entity_num, relation_num, d_in, d_model, d_out, n_layers, n_head):
        super().__init__()
        self.entity_embedding = nn.Embedding(entity_num, d_in)
        self.relation_embedding = nn.Embedding(relation_num * 2, d_in)
        self.input_fc = nn.Linear(d_in, d_model)
        self.gt_layers = nn.ModuleList([GTLayer(d_model, n_head) for _ in range(n_layers)])
        self.output_fc = nn.Linear(d_model, d_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, graph):
        x = self.entity_embedding(graph.entity)
        z = self.input_fc(x)
        embeds = self.dropout(z)
        for gt in self.gt_layers:
            embeds = gt(graph, embeds)
        return self.output_fc(embeds)


# ==================== 3. CLIP Text Encoder ====================
class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TextTransformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


# ==================== 4. CLIP Model ====================
class CLIPModel(nn.Module):
    def __init__(self, entity_num, relation_num, embed_dim, context_length,
                 vocab_size, t_width, t_layers, t_heads,
                 gnn_input, gnn_output, gt_layers, gt_head, neigh_num, lr):
        super().__init__()
        self.neigh_num = neigh_num
        self.gnn_output = gnn_output
        self.context_length = context_length

        self.gnn = GraphTransformer(entity_num, relation_num, gnn_input, gnn_input, gnn_output, gt_layers, gt_head)
        self.transformer = TextTransformer(t_width, t_layers, t_heads, attn_mask=self._build_mask(context_length))
        self.token_embedding = nn.Embedding(vocab_size, t_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, t_width))
        self.ln_final = LayerNorm(t_width)
        self.text_projection = nn.Parameter(torch.empty(t_width, embed_dim))

        self._init_params()
        self.optim = optim.Adam(self.parameters(), lr=lr)

    def _build_mask(self, ctx_len):
        mask = torch.empty(ctx_len, ctx_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def _init_params(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        nn.init.normal_(self.gnn.entity_embedding.weight, std=0.02)
        nn.init.normal_(self.gnn.relation_embedding.weight, std=0.02)

    def encode_graph(self, src, g):
        return self.gnn(g)[src]

    def encode_text(self, text):
        x = self.token_embedding(text).float()
        x = x + self.positional_embedding.float()
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).float()
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = x @ self.text_projection
        return x

    def align_loss(self, s, t, labels):
        logit_scale = (torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = logit_scale * s @ t.t()
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

    def align_pred(self, s, t):
        logit_scale = (torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = logit_scale * s @ t.t()
        return torch.argmax(logits, dim=1)

    def forward(self, g, src, rel, dst, src_text, dst_text, device):
        s_graph = self.encode_graph(src, g)
        s_text = self.encode_text(src_text)
        t_text = self.encode_text(dst_text)
        t_text = t_text.reshape(s_graph.shape[0], self.neigh_num, self.gnn_output)
        t_text = torch.mean(t_text, dim=1)
        s_graph = F.normalize(s_graph, dim=-1)
        s_text = F.normalize(s_text, dim=-1)
        t_text = F.normalize(t_text, dim=-1)
        labels = torch.arange(s_graph.shape[0]).to(device)
        return s_graph, s_text, t_text, labels


# ==================== 5. Data Loading ====================
def get_id_map(path):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            d[parts[0]] = int(parts[1])
    return d

def get_id2text(path):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            d[int(parts[0])] = parts[1] if len(parts) > 1 else ""
    return d

def load_triples(data_path, splits, ent2id, rel2id):
    src_list, dst_list, rel_list = [], [], []
    pos_tails = defaultdict(set)
    for split in splits:
        with open(os.path.join(data_path, f"{split}.txt")) as f:
            for line in f:
                s, r, d = line.strip().split('\t')
                s, r, d = ent2id[s], rel2id[r], ent2id[d]
                src_list.append(s)
                dst_list.append(d)
                rel_list.append(r)
                pos_tails[(s, r)].add(d)
    return src_list, dst_list, rel_list, pos_tails

def build_graph(src_list, dst_list, rel_list, entity_num, relation_num):
    src = torch.LongTensor(src_list)
    dst = torch.LongTensor(dst_list)
    rel = torch.LongTensor(rel_list)
    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    edge_index = torch.stack([src_all, dst_all])
    data = Data(edge_index=edge_index)
    data.entity = torch.arange(entity_num)
    return data


class TAGDataset(Dataset):
    def __init__(self, pos_tails, relation_num, neigh_num):
        self.query = []
        self.label = []
        self.neigh_num = neigh_num
        for k, v in pos_tails.items():
            self.query.append((k[0], k[1], -1))
            self.label.append(list(v))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        h, r, _ = self.query[idx]
        t = [choice(self.label[idx]) for _ in range(self.neigh_num)]
        return (h, r, t)

    @staticmethod
    def collate_fn(data):
        src = torch.tensor([d[0] for d in data], dtype=torch.long)
        rel = torch.tensor([d[1] for d in data], dtype=torch.long)
        dst = torch.tensor([d[2] for d in data], dtype=torch.long)
        return src, rel, dst


# ==================== 6. Train & Eval ====================
def train_one_epoch(model, loader, graph, id2text, device, ctx_len, edge_coef):
    model.train()
    total_loss = 0
    for step, (src, rel, dst) in enumerate(tqdm(loader, desc="  Training")):
        src_arr = src.numpy()
        dst_arr = dst.numpy().reshape(-1)
        src_texts = tokenize([id2text.get(i, "") for i in src_arr], context_length=ctx_len).to(device)
        dst_texts = tokenize([id2text.get(j, "") for j in dst_arr], context_length=ctx_len).to(device)
        src, rel, dst = src.to(device), rel.to(device), dst.to(device)

        s_graph, s_text, t_text, labels = model(graph, src, rel, dst, src_texts, dst_texts, device)
        loss1 = model.align_loss(s_graph, s_text, labels)
        loss2 = model.align_loss(s_graph, t_text, labels)
        loss3 = model.align_loss(s_text, t_text, labels)
        loss = loss1 + edge_coef * loss2 + edge_coef * loss3

        model.optim.zero_grad()
        loss.backward()
        model.optim.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, graph, id2text, device, ctx_len):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for src, rel, dst in tqdm(loader, desc="  Evaluating"):
            src_arr = src.numpy()
            dst_arr = dst.numpy().reshape(-1)
            src_texts = tokenize([id2text.get(i, "") for i in src_arr], context_length=ctx_len).to(device)
            dst_texts = tokenize([id2text.get(j, "") for j in dst_arr], context_length=ctx_len).to(device)
            src, rel, dst = src.to(device), rel.to(device), dst.to(device)

            s_graph, s_text, t_text, labels = model(graph, src, rel, dst, src_texts, dst_texts, device)
            pred1 = model.align_pred(s_graph, s_text)
            pred2 = model.align_pred(s_graph, t_text)
            true_labels = labels.cpu().numpy().tolist()
            all_true.extend(true_labels)
            all_true.extend(true_labels)
            all_pred.extend(pred1.cpu().numpy().tolist())
            all_pred.extend(pred2.cpu().numpy().tolist())
    return accuracy_score(all_true, all_pred) if all_true else 0.0


# ==================== 7. Main ====================
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # --- 小参数配置 (CPU 可跑) ---
    EMBED_DIM = 32
    T_WIDTH = 64
    T_LAYERS = 2
    T_HEADS = 4
    GNN_DIM = 32
    GT_LAYERS = 2
    GT_HEADS = 4
    CTX_LEN = 64
    VOCAB_SIZE = 49408
    BATCH_SIZE = 16
    EPOCHS = 2
    LR = 1e-4
    EDGE_COEF = 1.0
    NEIGH_NUM = 3
    MAX_TRAIN_SAMPLES = 500  # 只用前500个query加速
    MAX_EVAL_SAMPLES = 100

    data_path = os.path.join("aligner", "data", "FB15k-237N")
    device = torch.device("cpu")

    logging.info("Loading data...")
    ent2id = get_id_map(os.path.join(data_path, "mid2id.txt"))
    rel2id = get_id_map(os.path.join(data_path, "rel2id.txt"))
    id2text = get_id2text(os.path.join(data_path, "id2text.txt"))
    entity_num = len(ent2id)
    relation_num = len(rel2id)
    logging.info(f"  entities={entity_num}, relations={relation_num}")

    logging.info("Building graph...")
    src_all, dst_all, rel_all, _ = load_triples(data_path, ['train', 'valid', 'test'], ent2id, rel2id)
    graph = build_graph(src_all, dst_all, rel_all, entity_num, relation_num).to(device)
    logging.info(f"  nodes={entity_num}, edges={graph.edge_index.shape[1]}")

    logging.info("Preparing datasets...")
    _, _, _, pos_tails_train = load_triples(data_path, ['train', 'valid'], ent2id, rel2id)
    _, _, _, pos_tails_test = load_triples(data_path, ['test'], ent2id, rel2id)

    train_ds = TAGDataset(dict(list(pos_tails_train.items())[:MAX_TRAIN_SAMPLES]), relation_num, NEIGH_NUM)
    eval_ds = TAGDataset(dict(list(pos_tails_test.items())[:MAX_EVAL_SAMPLES]), relation_num, NEIGH_NUM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TAGDataset.collate_fn)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=TAGDataset.collate_fn)
    logging.info(f"  train queries={len(train_ds)}, eval queries={len(eval_ds)}")

    logging.info("Building CLIP model (small)...")
    model = CLIPModel(
        entity_num=entity_num, relation_num=relation_num,
        embed_dim=EMBED_DIM, context_length=CTX_LEN,
        vocab_size=VOCAB_SIZE, t_width=T_WIDTH, t_layers=T_LAYERS, t_heads=T_HEADS,
        gnn_input=GNN_DIM, gnn_output=EMBED_DIM, gt_layers=GT_LAYERS, gt_head=GT_HEADS,
        neigh_num=NEIGH_NUM, lr=LR
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"  Total parameters: {total_params:,}")

    save_dir = os.path.join("checkpoints_cpu", "FB15k-237N")
    os.makedirs(save_dir, exist_ok=True)

    logging.info("=" * 50)
    logging.info("Stage 1: Aligner Training (CPU Dry-Run)")
    logging.info("=" * 50)
    best_acc = 0
    for epoch in range(EPOCHS):
        logging.info(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        avg_loss = train_one_epoch(model, train_loader, graph, id2text, device, CTX_LEN, EDGE_COEF)
        logging.info(f"  Train loss: {avg_loss:.4f}")

        acc = evaluate(model, eval_loader, graph, id2text, device, CTX_LEN)
        logging.info(f"  Eval accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(save_dir, "aligner_best.pkl")
            torch.save(model.state_dict(), save_path)
            logging.info(f"  Saved best model -> {save_path}")

    # 保存实体嵌入
    logging.info("\nExporting entity embeddings...")
    emb_weight = model.gnn.entity_embedding.weight.data
    torch.save(emb_weight, os.path.join(save_dir, "entity_embedding.pt"))

    # 保存图数据 (供 Stage 2 使用)
    logging.info("Exporting graph data...")
    graph_with_feat = Data(x=emb_weight, edge_index=graph.edge_index)
    graph_with_feat.entity = graph.entity
    graph_data_all = {"FB15k-237N": graph_with_feat}
    torch.save(graph_data_all, os.path.join(save_dir, "graph_data_all.pt"))

    # 保存 config
    config = {
        "entity_num": entity_num, "relation_num": relation_num,
        "embed_dim": EMBED_DIM, "context_length": CTX_LEN,
        "vocab_size": VOCAB_SIZE, "transformer_width": T_WIDTH,
        "transformer_layers": T_LAYERS, "transformer_heads": T_HEADS,
        "gnn_type": "gt", "gnn_input": GNN_DIM, "gnn_hidden": GNN_DIM,
        "gnn_output": EMBED_DIM, "node_num": 1, "gt_layers": GT_LAYERS,
        "att_d_model": GNN_DIM, "gt_head": GT_HEADS,
        "att_norm": True, "if_pos": False,
        "edge_coef": EDGE_COEF, "lr": LR, "neigh_num": NEIGH_NUM
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logging.info(f"\n{'=' * 50}")
    logging.info(f"Stage 1 DONE! Best accuracy: {best_acc:.4f}")
    logging.info(f"Outputs saved to: {save_dir}/")
    logging.info(f"  - aligner_best.pkl       (model weights)")
    logging.info(f"  - entity_embedding.pt     (entity embeddings)")
    logging.info(f"  - graph_data_all.pt       (graph data for Stage 2)")
    logging.info(f"  - config.json             (model config)")
    logging.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
