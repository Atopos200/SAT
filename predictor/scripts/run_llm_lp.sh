prompt_mode=data_llm_lp
data_name=FB15k-237N

model_path=./Llama-2-7b-chat-hf
instruct_ds=./${prompt_mode}/${data_name}/train.json
graph_data_path=./clip/graph_data_all.pt
pretra_gnn=clip_gt_${data_name}
output_model=./checkpoints/${prompt_mode}/${data_name}

date_time=0109
GPU_DEVICE=1

wandb offline
CUDA_VISIBLE_DEVICES=$GPU_DEVICE nohup python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=20001 \
    ./train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --is_graph True > ./logs/${prompt_mode}/${data_name}/predicter_${date_time}.log 2>&1 &


# bash ./scripts/run_llm_lp.sh

