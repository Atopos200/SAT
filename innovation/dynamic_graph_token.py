"""
Dynamic Graph Tokenizer
将变长子图节点序列压缩为固定数量的 graph summary tokens。

使用 Perceiver 风格的 cross-attention：
  - M 个可学习 query tokens 作为"信息提取器"
  - 从 N 个子图节点特征中提取信息
  - 用 importance scores 加权注意力
  - 输出 M 个固定维度的 graph tokens
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from innovation.config import InnovationConfig


class ImportanceWeightedAttention(nn.Module):
    """带重要性权重的多头交叉注意力"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                importance: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: (B, M, D) - learnable query tokens
            key:   (B, N, D) - node features
            value: (B, N, D) - node features
            importance: (B, N) - importance scores from subgraph selector
        """
        B, M, _ = query.shape
        _, N, _ = key.shape

        q = self.q_proj(query).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if importance is not None:
            importance_bias = torch.log(importance.clamp(min=1e-6))
            importance_bias = importance_bias.unsqueeze(1).unsqueeze(2)
            attn = attn + importance_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, M, self.d_model)
        return self.out_proj(out)


class DynamicGraphTokenizer(nn.Module):
    """
    将变长子图压缩为固定数量的 graph tokens。

    Pipeline:
      1. node_features (N, D_in) -> input projection (N, D)
      2. importance scores (N,) 作为注意力偏置
      3. M 个可学习 query tokens cross-attend 到 N 个节点
      4. 经过 feed-forward 输出 M 个 graph tokens (M, D)
    """

    def __init__(self, config: InnovationConfig, input_dim: int = None):
        super().__init__()
        D = config.token_dim
        M = config.num_graph_tokens
        input_dim = input_dim or D

        self.query_tokens = nn.Parameter(torch.randn(1, M, D) * 0.02)
        self.input_proj = nn.Linear(input_dim, D) if input_dim != D else nn.Identity()
        self.cross_attn = ImportanceWeightedAttention(D, config.num_compress_heads, config.compress_dropout)
        self.norm1 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.GELU(),
            nn.Linear(D * 4, D),
        )
        self.norm2 = nn.LayerNorm(D)
        self.output_norm = nn.LayerNorm(D)

    def forward(self, node_features: torch.Tensor, importance_scores: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            node_features: (N, D_in) 子图节点特征
            importance_scores: (N,) 节点重要性分数
        Returns:
            graph_tokens: (M, D) 压缩后的图 tokens
        """
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        if importance_scores is not None and importance_scores.dim() == 1:
            importance_scores = importance_scores.unsqueeze(0)

        B = node_features.shape[0]
        x = self.input_proj(node_features)

        queries = self.query_tokens.expand(B, -1, -1)
        attended = self.cross_attn(queries, x, x, importance_scores)
        attended = self.norm1(queries + attended)

        ffn_out = self.ffn(attended)
        out = self.norm2(attended + ffn_out)
        out = self.output_norm(out)

        return out.squeeze(0)
