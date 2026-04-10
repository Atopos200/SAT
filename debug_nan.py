"""
诊断 loss=nan 的原因
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

DATA_PATH = "predictor/data_llm_lp/FB15k-237N/train.json"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B"
MAX_LEN = 384

print("=" * 50)
print("Step 1: 检查数据")
print("=" * 50)
with open(DATA_PATH) as f:
    data = json.load(f)[:10]

for i, item in enumerate(data):
    q = item['conversations'][0]['value'].replace("<graph>", "[GRAPH]")
    a = item['conversations'][1]['value']
    prompt = f"Question: {q}\nAnswer: {a}"
    print(f"样本 {i}: 问题长度={len(q.split())}词, 答案='{a[:40]}'")

print("\n" + "=" * 50)
print("Step 2: 检查 tokenizer")
print("=" * 50)
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"vocab_size={tokenizer.vocab_size}, pad_token='{tokenizer.pad_token}'")

# 检查一条样本的 token 化结果
item = data[0]
q = item['conversations'][0]['value'].replace("<graph>", "[GRAPH]")
a = item['conversations'][1]['value']
prompt = f"Question: {q}\nAnswer: {a}"
enc = tokenizer(prompt, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
input_ids = enc.input_ids[0]
print(f"input_ids shape: {input_ids.shape}")
print(f"非pad token数: {(input_ids != tokenizer.pad_token_id).sum().item()}")
print(f"input_ids 有 nan: {torch.isnan(input_ids.float()).any()}")

# 构造 labels
answer_ids = tokenizer(a, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
question_text = f"Question: {q}\nAnswer: "
q_max_len = MAX_LEN - len(answer_ids)
q_ids = tokenizer(question_text, add_special_tokens=True, max_length=q_max_len, truncation=True).input_ids
full_ids = q_ids + answer_ids
pad_id = tokenizer.pad_token_id
pad_len = MAX_LEN - len(full_ids)
full_ids = full_ids + [pad_id] * pad_len if pad_len > 0 else full_ids[:MAX_LEN]
input_ids = torch.tensor(full_ids, dtype=torch.long)
attention_mask = (input_ids != pad_id).long()
labels = torch.full_like(input_ids, -100)
ans_start = len(q_ids)
ans_end = min(ans_start + len(answer_ids), MAX_LEN)
labels[ans_start:ans_end] = input_ids[ans_start:ans_end]
labels[attention_mask == 0] = -100
valid_labels = (labels != -100).sum().item()
enc = type('obj', (object,), {'input_ids': input_ids.unsqueeze(0), 'attention_mask': attention_mask.unsqueeze(0)})()
print(f"有效 label 数: {valid_labels} (应该 > 0)")
print(f"答案 token 数: {len(answer_ids)}, 答案起始位置: {ans_start}")

print("\n" + "=" * 50)
print("Step 3: 检查模型一步前向传播")
print("=" * 50)
model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, trust_remote_code=True, dtype=torch.float32)
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.eval()

with torch.no_grad():
    out = model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=enc.attention_mask,
        labels=labels.unsqueeze(0)
    )
print(f"loss = {out.loss.item()}")
print(f"logits 有 nan: {torch.isnan(out.logits).any().item()}")
print(f"logits 有 inf: {torch.isinf(out.logits).any().item()}")

if torch.isnan(out.loss):
    print("\n⚠ loss 是 nan！原因分析:")
    print(f"  valid_labels = {valid_labels}")
    if valid_labels == 0:
        print("  → labels 全是 -100，没有有效监督信号，这是 nan 的原因！")
        print("  → 需要检查 prefix_len 计算是否超过了 MAX_LEN")
        print(f"  → prefix_len={prefix_len}, MAX_LEN={MAX_LEN}")
    else:
        print("  → labels 有效，可能是数值问题")
else:
    print("\n✓ 单步前向传播正常，loss 不是 nan")
    print("  nan 可能在多步训练后出现，原因是学习率或梯度爆炸")
