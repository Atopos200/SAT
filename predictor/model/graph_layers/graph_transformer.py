import torch
from torch import nn
import torch.nn.functional as F
import math
from transformers.configuration_utils import PretrainedConfig


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

def pos_encoding(pe, learn_pe, nvar, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((nvar, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        W_pos = torch.empty((nvar, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "zeros":
        W_pos = torch.empty((nvar, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "normal" or pe == "gauss":
        W_pos = torch.zeros((nvar, 1))
        nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        W_pos = torch.zeros((nvar, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == "sincos":
        W_pos = PositionalEncoding(nvar, d_model, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)"
        )
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class graph_transformer(nn.Module):
    def __init__(self, args):
        super(graph_transformer, self).__init__()
        self.config = PretrainedConfig()
        self.args = args

        self.entity_embedding = nn.Embedding(args.entity_num, args.gnn_input)
        self.relation_embedding = nn.Embedding(args.relation_num*2, args.gnn_input)

        self.gtLayers = nn.Sequential(*[GTLayer(args) for _ in range(args.gt_layers)])

        self.pos_enc = pos_encoding("zeros", True, args.node_num, args.att_d_model)
        self.input_fc = nn.Linear(args.gnn_input, args.att_d_model)
        self.dropout = nn.Dropout(0.1)
        self.output_fc = nn.Linear(args.att_d_model, args.gnn_output)

    def forward(self, graph):
        device = self.parameters().__next__().device
        graph = graph.to(device)

        x = graph.graph_node
        x, self.input_fc.weight, self.input_fc.bias, self.pos_enc = Mv2Samedevice([x, self.input_fc.weight, self.input_fc.bias, self.pos_enc])
        z = self.input_fc(x)
        if self.args.if_pos:
            embeds = self.dropout(z + self.pos_enc)
        else:
            embeds = self.dropout(z)
        for gt in self.gtLayers:
            embeds = gt(graph, embeds)
        embeds, self.output_fc.weight, self.output_fc.bias = Mv2Samedevice(
            [embeds, self.output_fc.weight, self.output_fc.bias]
        )
        ent_embeds = self.output_fc(embeds)

        return ent_embeds


def Mv2Samedevice(vars): 
    return [var.to(vars[0].device) for var in vars]


class GTLayer(nn.Module):
    def __init__(self, args):
        super(GTLayer, self).__init__()
        self.args = args
        self.qTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.att_d_model, args.att_d_model)))
        self.kTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.att_d_model, args.att_d_model)))
        self.vTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.att_d_model, args.att_d_model)))
        if args.att_norm:
            self.norm = nn.LayerNorm(args.att_d_model, eps=1e-6)

    def forward(self, g, embeds):
        rows, cols = g.edge_index
        nvar, _ = embeds.shape

        rowEmbeds = embeds[rows, :]
        colEmbeds = embeds[cols, :]
        evar, _ = rowEmbeds.shape

        qEmbeds = (rowEmbeds @ self.qTrans).view([evar, self.args.gt_head, self.args.att_d_model // self.args.gt_head])
        kEmbeds = (colEmbeds @ self.kTrans).view([evar, self.args.gt_head, self.args.att_d_model // self.args.gt_head])
        vEmbeds = (colEmbeds @ self.vTrans).view([evar, self.args.gt_head, self.args.att_d_model // self.args.gt_head])

        att = torch.einsum("ehd, ehd -> eh", qEmbeds, kEmbeds)
        att = torch.clamp(att, -10.0, 10.0)
        expAtt = torch.exp(att)

        tem = torch.zeros([nvar, self.args.gt_head]).to(expAtt.device, dtype=expAtt.dtype)
        rows = rows.to(expAtt.device)
        attNorm = (tem.index_add_(0, rows, expAtt))[rows, :]
        att = expAtt / (attNorm + 1e-8)

        resEmbeds = torch.einsum("eh, ehd -> ehd", att, vEmbeds).view([evar, self.args.att_d_model])
        tem = torch.zeros([nvar, self.args.att_d_model]).to(resEmbeds.device, dtype=resEmbeds.dtype)
        rows = rows.to(resEmbeds.device)
        resEmbeds = tem.index_add_(0, rows, resEmbeds)
        resEmbeds = resEmbeds + embeds
        if self.args.att_norm:
            resEmbeds = self.norm(resEmbeds)

        return resEmbeds
