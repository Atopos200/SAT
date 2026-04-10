import json
import os
from typing import Dict, Any

import torch
from tqdm import tqdm

from dsgr.train.checkpoint import get_trainable_state_dict, load_trainable_state_dict
from dsgr.train.evaluate import evaluate_ranking_dataset


def train_variant_with_eval(
    model_v,
    tokenizer,
    device,
    train_loader,
    valid_data,
    eval_kwargs: Dict[str, Any],
    variant_dir: str,
    variant_name: str,
    epochs: int,
    grad_accum: int,
    lr: float,
    resume: bool,
):
    os.makedirs(variant_dir, exist_ok=True)
    ckpt_last_lora = os.path.join(variant_dir, "last_lora.pt")
    ckpt_best_lora = os.path.join(variant_dir, "best_lora.pt")
    ckpt_train_state = os.path.join(variant_dir, "train_state.json")

    optimizer = torch.optim.AdamW([p for p in model_v.parameters() if p.requires_grad], lr=lr)
    start_epoch = 0
    best_valid_mrr = -1.0
    best_epoch = -1
    if resume and os.path.exists(ckpt_train_state):
        with open(ckpt_train_state, "r", encoding="utf-8") as f:
            st = json.load(f)
        if st.get("variant_name") == variant_name:
            start_epoch = int(st.get("next_epoch", 0))
            best_valid_mrr = float(st.get("best_valid_mrr", -1.0))
            best_epoch = int(st.get("best_epoch", -1))
            load_trainable_state_dict(model_v, ckpt_last_lora)

    model_v.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"  Epoch {epoch + 1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_v(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model_v.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
            total_loss += outputs.loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        model_v.eval()
        valid_metrics, _ = evaluate_ranking_dataset(
            model_v=model_v,
            tokenizer=tokenizer,
            device=device,
            eval_data=valid_data,
            **eval_kwargs,
        )
        valid_mrr = valid_metrics["MRR"]

        torch.save(get_trainable_state_dict(model_v), ckpt_last_lora)
        with open(ckpt_train_state, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "variant_name": variant_name,
                    "next_epoch": epoch + 1,
                    "best_valid_mrr": best_valid_mrr,
                    "best_epoch": best_epoch,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        if valid_mrr > best_valid_mrr:
            best_valid_mrr = valid_mrr
            best_epoch = epoch + 1
            torch.save(get_trainable_state_dict(model_v), ckpt_best_lora)
            with open(ckpt_train_state, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "variant_name": variant_name,
                        "next_epoch": epoch + 1,
                        "best_valid_mrr": best_valid_mrr,
                        "best_epoch": best_epoch,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        model_v.train()

    return {
        "avg_loss_last_epoch": avg_loss,
        "best_epoch": best_epoch,
        "best_valid_mrr": best_valid_mrr,
        "ckpt_last_lora": ckpt_last_lora,
        "ckpt_best_lora": ckpt_best_lora,
        "ckpt_train_state": ckpt_train_state,
    }

