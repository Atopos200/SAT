import torch
import torch.nn as nn


def score_candidate_entities(model_v, tokenizer, device, prompt, candidate_texts, max_len, batch_size=32):
    """Score candidate entity names by conditional likelihood (avg token logprob)."""
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    if len(prompt_ids) >= max_len:
        prompt_ids = prompt_ids[:max_len - 1]

    scored = []
    for i in range(0, len(candidate_texts), batch_size):
        chunk = candidate_texts[i:i + batch_size]
        full_batch, label_batch = [], []
        for cand in chunk:
            cand_ids = tokenizer(cand, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            avail = max_len - len(prompt_ids)
            if avail <= 0:
                continue
            cand_ids = cand_ids[:avail]
            full_ids = prompt_ids + cand_ids
            labels = [-100] * len(prompt_ids) + cand_ids
            full_batch.append(torch.tensor(full_ids, dtype=torch.long))
            label_batch.append(torch.tensor(labels, dtype=torch.long))

        if not full_batch:
            continue

        input_ids = nn.utils.rnn.pad_sequence(full_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = nn.utils.rnn.pad_sequence(label_batch, batch_first=True, padding_value=-100)
        attn_mask = (input_ids != tokenizer.pad_token_id).long()

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            out = model_v(input_ids=input_ids, attention_mask=attn_mask)
            logits = out.logits[:, :-1, :]
            target = labels[:, 1:]
            valid_mask = (target != -100)
            safe_target = target.masked_fill(~valid_mask, 0)
            log_probs = torch.log_softmax(logits, dim=-1).gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
            token_sum = (log_probs * valid_mask).sum(dim=-1)
            token_cnt = valid_mask.sum(dim=-1).clamp(min=1)
            score = token_sum / token_cnt

        scored.extend([float(s) for s in score.tolist()])
    return scored


def build_filtered_candidate_ids(
    item,
    h_id,
    r_id,
    t_id,
    true_tails_by_hr,
    tails_by_rel,
    all_entity_ids,
    mode="relation",
    max_candidates=0,
):
    """Build filtered ranking candidates while always preserving true tail."""
    if mode == "all":
        candidate_ids = set(all_entity_ids)
    elif mode == "subgraph":
        candidate_ids = set(item.get("graph", {}).get("node_list", []))
    else:
        candidate_ids = set(tails_by_rel.get(r_id, set()))
    candidate_ids.add(t_id)
    if not candidate_ids:
        return []

    hr_truth = true_tails_by_hr.get((h_id, r_id), set())
    filtered_ids = [cid for cid in candidate_ids if cid == t_id or cid not in hr_truth]
    if t_id not in filtered_ids:
        filtered_ids.append(t_id)

    if max_candidates > 0 and len(filtered_ids) > max_candidates:
        filtered_ids = sorted(filtered_ids)
        keep_n = max(max_candidates - 1, 1)
        filtered_ids = filtered_ids[:keep_n]
        if t_id not in filtered_ids:
            filtered_ids.append(t_id)

    return sorted(set(filtered_ids))


def compute_graph_support_scores(item, candidate_ids):
    """Estimate candidate support from retrieved local subgraph evidence."""
    graph = item.get("graph", {}) if isinstance(item, dict) else {}
    node_list = graph.get("node_list", []) or []
    importance_scores = graph.get("importance_scores", []) or []
    paths = graph.get("paths", []) or []

    importance_map = {}
    for nid, score in zip(node_list, importance_scores):
        try:
            importance_map[int(nid)] = float(score)
        except (TypeError, ValueError):
            continue

    path_endpoint_bonus = {}
    for path in paths:
        if not path:
            continue
        try:
            end_node = int(path[-1][-1])
        except (TypeError, ValueError, IndexError):
            continue
        bonus = 1.0 / max(len(path), 1)
        path_endpoint_bonus[end_node] = max(path_endpoint_bonus.get(end_node, 0.0), bonus)

    if importance_map:
        max_importance = max(max(importance_map.values()), 1e-6)
    else:
        max_importance = 1.0

    scores = []
    for cid in candidate_ids:
        imp = importance_map.get(cid, 0.0) / max_importance
        path_bonus = path_endpoint_bonus.get(cid, 0.0)
        support = 0.75 * imp + 0.25 * path_bonus
        scores.append(float(support))
    return scores


def evaluate_ranking_dataset(
    model_v,
    tokenizer,
    device,
    eval_data,
    true_tails_by_hr,
    tails_by_rel,
    all_entity_ids,
    entity_names,
    get_hrt_from_item,
    max_len,
    candidate_mode="relation",
    max_candidates=0,
    score_batch=32,
    graph_support_weight=0.15,
):
    hit1 = 0
    hit3 = 0
    hit10 = 0
    mrr_sum = 0.0
    total = 0
    skipped = 0
    predictions = []

    for item in eval_data:
        question = item["conversations"][0]["value"].replace("<graph>", "[GRAPH]")
        ground_truth = item["conversations"][1]["value"]
        hrt = get_hrt_from_item(item)
        if hrt is None:
            skipped += 1
            continue
        h_id, r_id, t_id = hrt

        candidate_ids_eval = build_filtered_candidate_ids(
            item=item,
            h_id=h_id,
            r_id=r_id,
            t_id=t_id,
            true_tails_by_hr=true_tails_by_hr,
            tails_by_rel=tails_by_rel,
            all_entity_ids=all_entity_ids,
            mode=candidate_mode,
            max_candidates=max_candidates,
        )
        if not candidate_ids_eval:
            skipped += 1
            continue
        candidate_texts = [entity_names.get(cid, f"Entity_{cid}") for cid in candidate_ids_eval]
        prompt = f"Question: {question}\nAnswer:"
        scores = score_candidate_entities(
            model_v=model_v,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            candidate_texts=candidate_texts,
            max_len=max_len,
            batch_size=score_batch,
        )
        if len(scores) != len(candidate_ids_eval):
            skipped += 1
            continue
        graph_support_scores = compute_graph_support_scores(item, candidate_ids_eval)
        if len(graph_support_scores) == len(scores):
            scores = [
                float(llm_s + graph_support_weight * graph_s)
                for llm_s, graph_s in zip(scores, graph_support_scores)
            ]

        ranked = sorted(zip(candidate_ids_eval, candidate_texts, scores), key=lambda x: x[2], reverse=True)
        rank = next((i + 1 for i, (cid, _, _) in enumerate(ranked) if cid == t_id), None)
        if rank is None:
            skipped += 1
            continue

        total += 1
        mrr_sum += 1.0 / rank
        if rank <= 1:
            hit1 += 1
        if rank <= 3:
            hit3 += 1
        if rank <= 10:
            hit10 += 1

        pred_answer = ranked[0][1]
        top5 = [
            {"entity_id": int(cid), "entity_name": txt, "score": round(float(sc), 4)}
            for cid, txt, sc in ranked[:5]
        ]
        predictions.append(
            {
                "id": item.get("id", ""),
                "ground_truth": ground_truth,
                "prediction": pred_answer,
                "rank": rank,
                "hit@1": 1 if rank <= 1 else 0,
                "hit@3": 1 if rank <= 3 else 0,
                "hit@10": 1 if rank <= 10 else 0,
                "candidate_count": len(candidate_ids_eval),
                "top5": top5,
            }
        )

    hit1_rate = hit1 / max(total, 1)
    hit3_rate = hit3 / max(total, 1)
    hit10_rate = hit10 / max(total, 1)
    mrr = mrr_sum / max(total, 1)
    return (
        {
            "Hit@1": round(hit1_rate * 100, 2),
            "Hit@3": round(hit3_rate * 100, 2),
            "Hit@10": round(hit10_rate * 100, 2),
            "MRR": round(mrr * 100, 2),
            "total": total,
            "hits@1": hit1,
            "hits@3": hit3,
            "hits@10": hit10,
            "skipped": skipped,
            "protocol": "filtered_ranking",
            "candidate_mode": candidate_mode,
            "max_candidates": max_candidates,
        },
        predictions,
    )

