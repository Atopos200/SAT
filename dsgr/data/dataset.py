"""Dataset definitions for DSGR training."""

import json

import torch
from torch.utils.data import Dataset


class KGCDataset(Dataset):
    """Knowledge graph completion dataset for causal LM training."""

    def __init__(self, data_path, tokenizer, max_len, max_samples, graph_data_all=None):
        with open(data_path) as f:
            self.data = json.load(f)[:max_samples]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.graph_data_all = graph_data_all

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["conversations"][0]["value"]
        answer = item["conversations"][1]["value"]
        question = question.replace("<graph>", "[GRAPH]")

        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        eos_id = self.tokenizer.eos_token_id
        answer_ids = answer_ids + [eos_id]

        question_text = f"Question: {question}\nAnswer: "
        q_max_len = self.max_len - len(answer_ids)
        q_ids = self.tokenizer(
            question_text,
            add_special_tokens=True,
            max_length=q_max_len,
            truncation=True,
        ).input_ids

        full_ids = q_ids + answer_ids
        pad_id = self.tokenizer.pad_token_id
        pad_len = self.max_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [pad_id] * pad_len
        else:
            full_ids = full_ids[:self.max_len]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = (input_ids != pad_id).long()

        labels = torch.full_like(input_ids, -100)
        ans_start = len(q_ids)
        ans_end = ans_start + len(answer_ids)
        if ans_end <= self.max_len:
            labels[ans_start:ans_end] = input_ids[ans_start:ans_end]
        else:
            labels[ans_start:self.max_len] = input_ids[ans_start:self.max_len]
        labels[attention_mask == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

