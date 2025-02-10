from statistics import mean
import torch
from torch import nn
import torch.nn.functional as F
import math


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
        self.args = args

        self.entity_embedding = nn.Embedding(args.entity_num, args.gnn_input)
        self.relation_embedding = nn.Embedding(args.relation_num*2, args.gnn_input)

        self.gtLayers = nn.Sequential(*[GTLayer(args) for _ in range(args.gt_layers)])

        self.pos_enc = pos_encoding("zeros", True, args.node_num, args.att_d_model)
        self.input_fc = nn.Linear(args.gnn_input, args.att_d_model)
        self.dropout = nn.Dropout(0.1)
        self.output_fc = nn.Linear(args.att_d_model, args.gnn_output)

    def forward(self, graph):
        x = self.entity_embedding(graph.entity)
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


class graph_transformer_kgc(nn.Module):
    def __init__(self, args):
        super(graph_transformer_kgc, self).__init__()
        self.args = args
        self.entity_embedding = nn.Embedding(args.entity_num, args.gnn_input)
        self.relation_embedding = nn.Embedding(args.relation_num*2, args.gnn_input)
        self.gtLayers = nn.Sequential(*[GTLayer(args) for _ in range(args.gt_layers)])

        self.pos_enc = pos_encoding("zeros", True, args.node_num, args.att_d_model)
        self.input_fc = nn.Linear(args.gnn_input, args.att_d_model)
        self.dropout = nn.Dropout(0.1)
        self.output_fc = nn.Linear(args.att_d_model, args.gnn_output)
        self.decoder = ConvE(args)


    def forward_kgc(self, src, rel, g):
        x = self.entity_embedding(g.entity)
        x, self.input_fc.weight, self.input_fc.bias, self.pos_enc = Mv2Samedevice([x, self.input_fc.weight, self.input_fc.bias, self.pos_enc])
        z = self.input_fc(x)
        if self.args.if_pos:
            embeds = self.dropout(z + self.pos_enc)
        else:
            embeds = self.dropout(z)
        for gt in self.gtLayers:
            embeds = gt(g, embeds)
        embeds, self.output_fc.weight, self.output_fc.bias = Mv2Samedevice(
            [embeds, self.output_fc.weight, self.output_fc.bias]
        )
        ent_embeds = self.output_fc(embeds)

        src_embeds = ent_embeds[src]
        rel_embeds = self.relation_embedding(rel)

        score = self.decoder(src_embeds, rel_embeds, ent_embeds)

        return ent_embeds, score


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

        rowEmbeds, self.qTrans, self.kTrans, self.vTrans = Mv2Samedevice(
            [rowEmbeds, self.qTrans, self.kTrans, self.vTrans]
        )
        qEmbeds = (rowEmbeds @ self.qTrans).view([evar, self.args.gt_head, self.args.att_d_model // self.args.gt_head])
        kEmbeds = (colEmbeds @ self.kTrans).view([evar, self.args.gt_head, self.args.att_d_model // self.args.gt_head])
        vEmbeds = (colEmbeds @ self.vTrans).view([evar, self.args.gt_head, self.args.att_d_model // self.args.gt_head])

        att = torch.einsum("ehd, ehd -> eh", qEmbeds, kEmbeds)
        att = torch.clamp(att, -10.0, 10.0)
        expAtt = torch.exp(att)

        tem = torch.zeros([nvar, self.args.gt_head]).to(expAtt.device)
        rows = rows.to(expAtt.device)
        attNorm = (tem.index_add_(0, rows, expAtt))[rows, :]
        att = expAtt / (attNorm + 1e-8)

        resEmbeds = torch.einsum("eh, ehd -> ehd", att, vEmbeds).view([evar, self.args.att_d_model])
        tem = torch.zeros([nvar, self.args.att_d_model]).to(resEmbeds.device)
        rows = rows.to(resEmbeds.device)
        resEmbeds = tem.index_add_(0, rows, resEmbeds)
        resEmbeds = resEmbeds + embeds
        if self.args.att_norm:
            resEmbeds, self.norm.weight, self.norm.bias = Mv2Samedevice([resEmbeds, self.norm.weight, self.norm.bias])
            resEmbeds = self.norm(resEmbeds)

        return resEmbeds


class ConvE(nn.Module):
    def __init__(self, args): 
        super().__init__()
        self.args = args

        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(args.out_channels)
        self.bn2 = nn.BatchNorm1d(args.gnn_output)

        self.conv_drop = nn.Dropout(0.3)
        self.fc_drop = nn.Dropout(0.3)
        assert args.ker_height * args.ker_width == args.gnn_output
        self.conv = nn.Conv2d(1, out_channels=args.out_channels, stride=1, padding=0,
                                    kernel_size=args.ker_size, bias=False)
        flat_size_height = int(2 * args.ker_height) - args.ker_size + 1
        flat_size_width = args.ker_width - args.ker_size + 1
        self.flat_size = flat_size_height * flat_size_width * args.out_channels
        self.fc = nn.Linear(self.flat_size, args.gnn_output, bias=False)
        self.ent_drop = nn.Dropout(0.3)

    def forward(self, head, rel, all_ent):
        c_head = head.reshape(-1, 1, head.shape[-1])
        c_rel = rel.reshape(-1, 1, rel.shape[-1])
        c_emb = torch.cat((c_head, c_rel), 1)
        c_emb = torch.transpose(c_emb, 2, 1).reshape((-1, 1, 2 * self.args.ker_height, self.args.ker_width))

        x = self.bn0(c_emb)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)
        x = x.view(-1, self.flat_size)
        x = self.fc(x)
        # x = self.fc_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc_drop(x)
        all_ent = self.ent_drop(all_ent)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x = torch.sigmoid(x)
        return x
