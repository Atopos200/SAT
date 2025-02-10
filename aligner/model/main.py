import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

import argparse
import random
import math
import time
import json

import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
from model_gt import CLIP, tokenize

from data_helper import get_mid2id, get_rel2id, get_id2text, load_data
from data_helper import construct_graph, save_clip_data
from data_helper import TAGTrainDataset, TrainDataset, EvalDataset
from data_helper import extract_negative_triples
from sklearn.metrics import accuracy_score


def train():
    model.train()
    best_test_acc = 0
    for epoch in range(0, args.epoch_num):
        epoch_loss = 0.0
        
        for step, batch in tqdm(enumerate(train_loader), disable=False, total=len(train_loader)):
            src, rel, dst = batch[0]
            gnn_labels = batch[1]

            src_arr = src.numpy()
            dst_arr = dst.numpy().reshape(-1)
            src_text, dst_text = [id2text[i] for i in src_arr], [id2text[j] for j in dst_arr]
            src_text = tokenize(src_text, context_length=args.context_length).to(device) # (B,L)
            dst_text = tokenize(dst_text, context_length=args.context_length).to(device) # (B*neigh_num,L)

            src, rel, dst = src.to(device), rel.to(device), dst.to(device) 
            gnn_labels = gnn_labels.to(device)    

            s_graph_feats, s_text_feats, t_text_feats, text_labels = model(
                whole_graph, src, rel, dst, src_text, dst_text, device
            )   

            s_node_loss = model.align_loss(s_graph_feats, s_text_feats, text_labels)
            s_gt_loss = model.align_loss(s_graph_feats, t_text_feats, text_labels)
            tt_loss = model.align_loss(s_text_feats, t_text_feats, text_labels)

            all_loss = s_node_loss + args.edge_coef * s_gt_loss + args.edge_coef * tt_loss
            # all_loss = s_node_loss + s_gt_loss + tt_loss

            model.optim.zero_grad()
            torch.cuda.empty_cache()
            all_loss.backward()
            model.optim.step()
            loss = round((all_loss.detach().clone()).cpu().item(), 4)
            if step % 100 == 0:
                logging.info("{}th loss in {} epoch:{}".format(step, epoch, loss))
            epoch_loss += loss / len(train_loader)
        logging.info("{}th epoch mean loss:{}".format(epoch, epoch_loss))
        torch.save(model.state_dict(), model_save_path.replace(".pkl",f"_{epoch}th.pkl"))

        test_acc = evaluate(epoch)
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            logging.info("{}th epoch save the best model".format(epoch))
            torch.save(model.state_dict(), model_save_path.replace(".pkl","_best.pkl"))            


def evaluate(epoch=0):
    model.eval()
    all_true, all_pred = [], []
    for step, batch in tqdm(enumerate(eval_loader), disable=False, total=len(eval_loader)):
        src, rel, dst = batch[0]
        gnn_labels = batch[1]

        src_arr = src.numpy()
        dst_arr = dst.numpy().reshape(-1)
        src_text, dst_text = [id2text[i] for i in src_arr], [id2text[j] for j in dst_arr]
        src_text = tokenize(src_text, context_length=args.context_length).to(device)
        dst_text = tokenize(dst_text, context_length=args.context_length).to(device)

        src, rel, dst = src.to(device), rel.to(device), dst.to(device) 
        gnn_labels = gnn_labels.to(device)    

        s_graph_feats, s_text_feats, t_text_feats, text_labels = model(
            whole_graph, src, rel, dst, src_text, dst_text, device
        )       

        s_node_pred = model.align_pred(s_graph_feats, s_text_feats, text_labels)
        s_gt_pred = model.align_pred(s_graph_feats, t_text_feats, text_labels)
        tt_pred = model.align_pred(s_text_feats, t_text_feats, text_labels)

        true_label = text_labels.cpu().numpy().tolist()
        s_node_pred = s_node_pred.cpu().detach().numpy().tolist()
        s_gt_pred = s_gt_pred.cpu().detach().numpy().tolist()
        tt_pred = tt_pred.cpu().detach().numpy().tolist()
        all_true.extend(true_label)
        all_true.extend(true_label)
        all_true.extend(true_label)
        all_pred.extend(s_node_pred)
        all_pred.extend(s_gt_pred)
        all_pred.extend(tt_pred)

    acc = accuracy_score(all_true, all_pred)
    logging.info("{}th epoch test accuracy:{:.4f}".format(epoch, acc))
    return acc
    

def train_kgc():
    model.train()

    for epoch in range(args.epoch_num):
        epoch_loss = 0.0
        
        for step, batch in tqdm(enumerate(train_loader), disable=False, total=len(train_loader)):
            src, rel, dst = batch[0]
            gnn_labels = batch[1]

            src_arr = src.numpy()
            dst_arr = dst.numpy().reshape(-1)
            src_text, dst_text = [id2text[i] for i in src_arr], [id2text[j] for j in dst_arr]
            src_text = tokenize(src_text, context_length=args.context_length).to(device)
            dst_text = tokenize(dst_text, context_length=args.context_length).to(device)

            src, rel, dst = src.to(device), rel.to(device), dst.to(device) 
            gnn_labels = gnn_labels.to(device)    

            s_graph_feats, t_graph_feats, s_text_feats, t_text_feats, gnn_logits, text_labels = model(
                whole_graph, src, rel, dst, src_text, dst_text, device
            )

            s_node_loss = model.align_loss(s_graph_feats, s_text_feats, text_labels)
            s_gt_loss = model.align_loss(s_graph_feats, t_text_feats, text_labels)
            t_node_loss = model.align_loss(t_graph_feats, t_text_feats, text_labels)
            t_gt_loss = model.align_loss(t_graph_feats, s_text_feats, text_labels)
            tt_loss = model.align_loss(s_text_feats, t_text_feats, text_labels)

            gnn_loss = model.gnn_loss(gnn_logits, gnn_labels)
            all_loss = s_node_loss + args.edge_coef * s_gt_loss + args.edge_coef * tt_loss + gnn_loss

            model.optim.zero_grad()
            torch.cuda.empty_cache()
            all_loss.backward()
            model.optim.step()
            loss = round((all_loss.detach().clone()).cpu().item(), 4)

            if step % 100 == 0:
                logging.info("{}th loss in {} epoch:{}".format(step, epoch + 1, loss))
            epoch_loss += loss / len(train_loader)
        logging.info("{}th epoch mean loss:{}".format(epoch + 1, epoch_loss))
        torch.save(model.state_dict(), model_save_path)


def save_llm_clip(args):
    model_path = f"{args.output_path}/{args.data_name}/gt-xxx-og_best.pkl"
    state = torch.load(model_path) 
    model.load_state_dict(state)

    entity_embedding = model.gnn.entity_embedding.cpu()
    embs = []
    for i in range(args.entity_num):
        i = torch.tensor([i])
        tmp = entity_embedding(i)
        embs.append(tmp)
    embs = torch.stack(embs, dim=0)
    embs = torch.squeeze(embs, dim=1)
    torch.save(embs, f"{args.output_path}/{args.data_name}/entity_embedding.pt")

    save_clip_data(args, data_flag=['train', 'valid', 'test'])

def assure_dir(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_logger(log_file='./log.txt'):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S')
     
    handler1 = logging.StreamHandler()
    handler1.setLevel(logging.INFO)
    handler1.setFormatter(format)
    logger.addHandler(handler1)
    
    if log_file:
        handler2 = logging.FileHandler(log_file)
        handler2.setLevel(logging.INFO)
        handler2.setFormatter(format)
        logger.addHandler(handler2)
    return logger


def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, default="./data", help="data path")
    parser.add_argument("--output_path", type=str, default="./checkpoints")
    parser.add_argument("--data_name", type=str, default="YAGO3-10")              
    parser.add_argument("--cpu_worker_num", type=int, default=3) 
    parser.add_argument("--label_smooth", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0) 

    parser.add_argument("--aggregation_times", type=int, default=2, help="Aggregation times")
    parser.add_argument("--epoch_num", type=int, default=100, help="epoch number")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--edge_coef", type=float, default=10)
    parser.add_argument("--neigh_num", type=int, default=3)

    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=12)
    parser.add_argument("--transformer_width", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=49408)

    # gt config
    parser.add_argument("--gnn_type", type=str, default="gt")
    parser.add_argument("--gnn_input", type=int, default=128)
    parser.add_argument("--gnn_hidden", type=int, default=128)
    parser.add_argument("--gnn_output", type=int, default=128)

    parser.add_argument("--node_num", type=int, default=1)
    parser.add_argument("--gt_layers", type=int, default=3)
    parser.add_argument("--att_d_model", type=int, default=128)
    parser.add_argument("--gt_head", type=int, default=8)
    parser.add_argument("--att_norm", type=bool, default=True)
    parser.add_argument("--if_pos", type=bool, default=False)

    # ConvE
    parser.add_argument("--out_channels", type=int, default=200)
    parser.add_argument("--ker_size", type=int, default=4)
    parser.add_argument("--ker_height", type=int, default=8)
    parser.add_argument("--ker_width", type=int, default=16)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parser()
    args.data_path = os.path.join(args.data_path, args.data_name)
    setup_seed(seed=1)
    args.cur_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_save_path = f'./logs/{args.data_name}/aligner_{args.cur_time}.log'
    logger = set_logger(log_save_path)
    logging.info(f"log file: {log_save_path}")
    logging.info(args)

    model_save_name = f"{args.data_name}/{args.gnn_type}-{args.cur_time}-og.pkl"
    model_save_path = os.path.join(args.output_path, model_save_name)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    id2text = get_id2text(os.path.join(args.data_path, "id2text.txt"))
    ent2id = get_mid2id(os.path.join(args.data_path, "mid2id.txt"))
    rel2id = get_rel2id(os.path.join(args.data_path, "rel2id.txt"))
    args.entity_num = len(ent2id)
    args.relation_num = len(rel2id)
    logging.info(f"entity_num: {args.entity_num}, relation_num: {args.relation_num}")

    # for split in ['train', 'valid', 'test']:
    #     extract_negative_triples(args, data_flag=[split])

    whole_graph = construct_graph(args, data_flag=['train', 'valid', 'test'])
    whole_graph = whole_graph.to(device)

    train_dataset = TAGTrainDataset(args, ['train', 'valid'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.cpu_worker_num,
        collate_fn=train_dataset.collate_fn
    )

    eval_dataset = TAGTrainDataset(args, ['test'])
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.cpu_worker_num,
        collate_fn=eval_dataset.collate_fn
    )

    model = CLIP(args).to(device)

    # model_path = "./checkpoints/FB15k-237N/gt-20250113_055751-og_0th.pkl"
    # state = torch.load(model_path) 
    # model.load_state_dict(state)

    train()
    logging.info("train success...")

