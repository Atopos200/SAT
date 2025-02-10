from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Union, List
from simple_tokenizer import SimpleTokenizer
from torch import nn, optim
from graph_transformer import graph_transformer

_tokenizer = SimpleTokenizer()


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, num_head, attn_mask=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, num_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) 
        else:
            self.attn_mask = None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.context_length = args.context_length
        self.edge_coef = args.edge_coef

        if args.gnn_type == "gt":
            self.gnn = graph_transformer(args)
        self.transformer = Transformer(
            width=args.transformer_width,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = args.vocab_size
        self.token_embedding = nn.Embedding(args.vocab_size, args.transformer_width)  # the embedding for all possible tokens
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, args.transformer_width)) # (128,512)
        self.ln_final = LayerNorm(args.transformer_width)

        self.text_projection = nn.Parameter(torch.empty(args.transformer_width, args.embed_dim)) #(512,128)

        if args.gnn_type == "gcn":
            self.dtype = self.gnn.vars[0].dtype
        elif args.gnn_type == "gt":
            self.dtype = self.gnn.pos_enc.dtype

        self.optim = optim.Adam(
            [
                {"params": self.token_embedding.weight},
                {"params": self.positional_embedding},
                {"params": self.transformer.parameters()},
                {"params": self.text_projection},
                {"params": self.gnn.parameters()},
            ],
            lr=args.lr,
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

        nn.init.normal_(self.gnn.entity_embedding.weight, std=0.02)
        nn.init.normal_(self.gnn.relation_embedding.weight, std=0.02)
        # nn.init.xavier_uniform_(self.entity_embedding)
        # nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf")) # 全部填充为-inf
        mask.triu_(1)  # zero out the lower diagonal 上三角矩阵，下三角置为0
        return mask

    def encode_graph(self, src, g):
        ent_embeds = self.gnn(g)
        src_embeds = ent_embeds[src]
        return src_embeds

    def encode_graph_kgc(self, src, rel, dst, g):
        ent_embeds, score = self.gnn(src, rel, g)
        src_embeds = ent_embeds[src]
        dst = dst.reshape(-1) # (B*3, )
        dst_embeds = ent_embeds[dst]

        return src_embeds, dst_embeds, score

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # (B, L, D) (64,128,512)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(
            1, 0, 2
        )  # NLD -> LND, batch_size * context_length *emb_dim -> context_length * batch_size  *emb_dim
        x = self.transformer(x)
        x = x.permute(
            1, 0, 2
        )  # LND -> NLD, context_length * batch_size *emb_dim -> batch_size * context_length *emb_dim
        x = self.ln_final(x).type(self.dtype) # (B,L,D) (64,128,512)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot （end of token） embedding (eot_token is the highest number in each sequence)
        # so there is node need to shorten the context length
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  #(B,D) (64, 512)
        x = x @ self.text_projection # (B,D) (B, 128)
        return x

    def align_loss(self, s_feats, t_feats, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = logit_scale * s_feats @ t_feats.t() # (B,B) 
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        ret_loss = (loss_i + loss_t) / 2
        return ret_loss
    
    def align_pred(self, s_feats, t_feats, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = logit_scale * s_feats @ t_feats.t() 
        pred = torch.argmax(logits, dim=1)
        return  pred
  
    def gnn_loss(self, logits, labels):
        # (bs, n_ent)
        self.bce = nn.BCELoss()
        loss = self.bce(logits, labels)
        return loss    

    def forward(self, g, src, rel, dst, src_text, dst_text, device):
        s_graph_feats = self.encode_graph(src, g) # (B,D_gnn) (64, 128)
        # s_graph_feats, t_graph_feats, gnn_logits = self.encode_graph_kgc(src, rel, dst, g) # (B,D_gnn) (64, 128)

        s_text_feats = self.encode_text(src_text) # (B,D_text)

        t_text_feats = self.encode_text(dst_text) #(B*3, D_text)

        t_text_feats = t_text_feats.reshape(s_graph_feats.shape[0], self.args.neigh_num, self.args.gnn_output)
        t_text_feats = torch.mean(t_text_feats, dim=1, keepdim=False) # (B,D)
        # normalized features
        s_graph_feats = s_graph_feats / s_graph_feats.norm(dim=-1, keepdim=True)

        s_text_feats = s_text_feats / s_text_feats.norm(dim=-1, keepdim=True)
        t_text_feats = t_text_feats / t_text_feats.norm(dim=-1, keepdim=True)
        text_labels = torch.arange(s_graph_feats.shape[0]).to(device) # (B,)

        # t_graph_feats = t_graph_feats.reshape(s_graph_feats.shape[0], self.args.neigh_num, self.args.gnn_output)
        # t_graph_feats = torch.mean(t_graph_feats, dim=1, keepdim=False) # (B,D)
        # t_graph_feats = t_graph_feats / t_graph_feats.norm(dim=-1, keepdim=True)
        
        # return s_graph_feats, t_graph_feats, s_text_feats, t_text_feats, gnn_logits, text_labels
    
        return s_graph_feats, s_text_feats, t_text_feats, text_labels


def tokenize(texts: Union[str, List[str]], context_length: int = 128, truncate: bool = True) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long) # (B, 128)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result
