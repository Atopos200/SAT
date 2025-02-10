import os
import math
import logging

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from itertools import chain
from random import choice
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import Data
from torch.utils.data import Dataset


def get_mid2id(input_path):
    mid2id_dict = {}
    with open(input_path, "r") as file:
        for line in file.readlines():
            mid, id = line.strip().split('\t')
            mid2id_dict[mid] = int(id)
    return mid2id_dict


def get_rel2id(input_path):
    rel2id_dict = {}
    with open(input_path, "r") as file:
        for line in file.readlines():
            rel, id = line.strip().split('\t')
            rel2id_dict[rel] = int(id)
    return rel2id_dict


def get_id2text(input_path):
    id2text_dict = {}
    with open(input_path, "r") as file:
        for line in file.readlines():    
            id, text = line.strip().split('\t')
            id2text_dict[int(id)] = text
    return id2text_dict  


def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]
    return edge_norm


def load_data(input_path, data_flag='train'):
    assert data_flag in [
        ['train'], ['valid'], ['test'],
        ['train', 'valid'], ['train', 'valid', 'test']
    ]
    ent2id = get_mid2id(os.path.join(input_path, "mid2id.txt"))
    rel2id = get_rel2id(os.path.join(input_path, "rel2id.txt"))

    if data_flag in [['train'], ['valid'], ['test']]:
        path = os.path.join(input_path, data_flag[0]+'.txt')
        file = open(path, encoding='utf-8')
    elif data_flag == ['train', 'valid']:
        path1 = os.path.join(input_path, 'train.txt')
        path2 = os.path.join(input_path, 'valid.txt')
        file1 = open(path1, encoding='utf-8')
        file2 = open(path2, encoding='utf-8')
        file = chain(file1, file2)
    elif data_flag == ['train', 'valid', 'test']:
        path1 = os.path.join(input_path, 'train.txt')
        path2 = os.path.join(input_path, 'valid.txt')
        path3 = os.path.join(input_path, 'test.txt')
        file1 = open(path1, encoding='utf-8')
        file2 = open(path2, encoding='utf-8')
        file3 = open(path3, encoding='utf-8')
        file = chain(file1, file2, file3)
    else:
        raise NotImplementedError

    src_list, dst_list, rel_list, triple_list = [], [], [], []
    pos_tails = defaultdict(set)
    pos_heads = defaultdict(set)
    pos_rels = defaultdict(set)

    for i, line in enumerate(file):
        src, rel, dst = line.strip().split('\t')
        src, rel, dst = ent2id[src], rel2id[rel], ent2id[dst]

        src_list.append(src)
        dst_list.append(dst)
        rel_list.append(rel)
        triple_list.append((src, rel, dst))

        pos_tails[(src, rel)].add(dst)
        pos_heads[(rel, dst)].add(src)
        pos_rels[(src, dst)].add(rel)

    output_dict = {
        'src_list': src_list,
        'dst_list': dst_list,
        'rel_list': rel_list,
        'triple_list': triple_list,
        'pos_tails': pos_tails,
        'pos_heads': pos_heads,
        'pos_rels': pos_rels
    }

    logging.info('num_entity: {}'.format(len(ent2id)))
    logging.info('num_relation: {}'.format(len(rel2id)))
    logging.info('num_triples: {}'.format(len(triple_list)))

    return output_dict


def construct_graph(args, data_flag='train'):
    logging.info("construct graph: {}".format(data_flag))
    data = load_data(args.data_path, data_flag)
    src = torch.LongTensor(data['src_list'])
    rel = torch.LongTensor(data['rel_list'])
    dst = torch.LongTensor(data['dst_list'])

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + args.relation_num))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(np.arange(args.entity_num))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, args.entity_num, args.relation_num)

    return data


def save_clip_data(args, data_flag='train'):
    data = load_data(args.data_path, data_flag)
    src = torch.LongTensor(data['src_list'])
    rel = torch.LongTensor(data['rel_list'])
    dst = torch.LongTensor(data['dst_list'])

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + args.relation_num))

    edge_index = torch.stack((src, dst))
    edge_type = rel
    node_feat = torch.load(f"{args.output_path}/{args.data_name}/entity_embedding.pt")

    data = Data(x=node_feat, edge_index = edge_index)
    data.entity = torch.from_numpy(np.arange(args.entity_num))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, args.entity_num, args.relation_num)

    graph_data_all = {f"{args.data_name}": data}
    torch.save(graph_data_all, f"{args.output_path}/{args.data_name}/graph_data.pt")



class TAGTrainDataset(Dataset):
    def __init__(self, args, data_flag):
        # self.triples = triples
        
        logging.info("load data: {}".format(data_flag))
        self.args = args
        self.data = load_data(args.data_path, data_flag)

        self.query = list()
        self.label = list()
        for k, v in self.data['pos_tails'].items():
            self.query.append((k[0], k[1], -1))
            self.label.append(list(v))
        for k, v in self.data['pos_heads'].items():
            self.query.append((k[1], k[0] + self.args.relation_num, -1))
            self.label.append(list(v))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        h, r, t = self.query[idx]
        t = [choice(self.label[idx]) for _ in range(self.args.neigh_num)]

        label = self.get_onehot_label(self.label[idx])

        return (h, r, t), label

    def get_onehot_label(self, label):
        onehot_label = torch.zeros(self.args.entity_num)
        onehot_label[label] = 1
        if self.args.label_smooth != 0.0:
            onehot_label = (1.0 - self.args.label_smooth) * onehot_label + (1.0 / self.args.entity_num)

        return onehot_label

    @staticmethod
    def collate_fn(data):
        src = np.array([d[0][0] for d in data])
        rel = np.array([d[0][1] for d in data])
        dst = np.array([d[0][2] for d in data])
        label = [d[1] for d in data]

        src = torch.from_numpy(src).to(dtype=torch.int64)
        rel = torch.from_numpy(rel).to(dtype=torch.int64)
        dst = torch.from_numpy(dst).to(dtype=torch.int64)
        label = torch.stack(label, dim=0)
        return (src, rel, dst), label




class TrainDataset(Dataset):
    def __init__(self, args, data_flag):
        self.args = args
        self.data = load_data(args.data_path, data_flag)
        self.query = []
        self.label = []
        for k, v in self.data['pos_tails'].items():
            self.query.append((k[0], k[1], -1))
            self.label.append(list(v))
        for k, v in self.data['pos_heads'].items():
            self.query.append((k[1], k[0] + self.args.relation_num, -1))
            self.label.append(list(v))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        h, r, t = self.query[idx]
        label = self.get_onehot_label(self.label[idx])

        return (h, r, t), label

    def get_onehot_label(self, label):
        onehot_label = torch.zeros(self.args.entity_num)
        onehot_label[label] = 1
        if self.args.label_smooth != 0.0:
            onehot_label = (1.0 - self.args.label_smooth) * onehot_label + (1.0 / self.n_ent)

        return onehot_label

    @staticmethod
    def collate_fn(data):
        src = [d[0][0] for d in data]
        rel = [d[0][1] for d in data]
        dst = [d[0][2] for d in data]
        label = [d[1] for d in data]

        src = torch.tensor(src, dtype=torch.int64)
        rel = torch.tensor(rel, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)
        label = torch.stack(label, dim=0)

        return (src, rel, dst), label


class EvalDataset(Dataset):
    def __init__(self, args, data_flag, mode):
        assert mode in ['head_batch', 'tail_batch']
        self.args = args
        self.mode = mode
        self.data = load_data(args.data_path, data_flag)
        self.triples = [_ for _ in zip(self.data['src_list'], self.data['rel_list'], self.data['dst_list'])]
        self.data_all = load_data(args.data_path, ['train', 'valid', 'test'])
        self.pos_t = self.data_all['pos_tails']
        self.pos_h = self.data_all['pos_heads']

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]

        if self.mode == 'tail_batch':
            # filter_bias
            filter_bias = np.zeros(self.args.entity_num, dtype=np.float)
            filter_bias[list(self.pos_t[(h, r)])] = -float('inf')
            filter_bias[t] = 0.
        elif self.mode == 'head_batch':
            # filter_bias
            filter_bias = np.zeros(self.args.entity_num, dtype=np.float)
            filter_bias[list(self.pos_h[(r, t)])] = -float('inf')
            filter_bias[h] = 0.
            h, r, t = t, r+self.args.relation_num, h
        else:
            raise NotImplementedError

        return (h, r, t), filter_bias.tolist(), self.mode

    @staticmethod
    def collate_fn(data):
        h = [d[0][0] for d in data]
        r = [d[0][1] for d in data]
        t = [d[0][2] for d in data]
        filter_bias = [d[1] for d in data]
        mode = data[0][-1]

        h = torch.tensor(h, dtype=torch.int64)
        r = torch.tensor(r, dtype=torch.int64)
        t = torch.tensor(t, dtype=torch.int64)
        filter_bias = torch.tensor(filter_bias, dtype=torch.float)

        return (h, r, t), filter_bias, mode

def extract_negative_triples(args, data_flag):     
    logging.info("load data: {}".format(data_flag))
    data = load_data(args.data_path, data_flag)

    entities = np.arange(args.entity_num, dtype=np.int32)
    neg_num = 2
    query = list()
    label = list()
    for k, v in data['pos_tails'].items():
        query.append((k[0], k[1], -1))
        label.append(list(v))

    def get_neg_ent(triple, label):
        mask = np.ones([args.entity_num], dtype=np.bool_)
        mask[label] = 0
        neg_ent = np.int32(np.random.choice(entities[mask], neg_num, replace=False))    
        return neg_ent
    output_handler = open(os.path.join(args.data_path, data_flag[0]+"_neg.txt"), "w")
    logging.info("Total number: {}".format(len(query)))
    
    for i in range(len(query)):
        
        cur_tripe, cur_label = query[i], label[i]
        neg_ents = get_neg_ent(cur_tripe, cur_label)
        neg_triples = []
        for neg_ent in neg_ents:
            h, r, t = cur_tripe[0], cur_tripe[1], neg_ent
            cur_tripe = [str(h), str(r), str(t)]
            neg_triples.append("\t".join(cur_tripe))
        
        output_handler.write("\n".join(neg_triples))
        output_handler.write("\n")
        
        if i % 1000 == 0:
            logging.info("finish {}".format(i))
    logging.info("finish all")

