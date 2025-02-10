prompt_mode=data_llm_lp
data_name=FB15k-237N
end_id=10

date_time=0109
GPU_DEVICE=4

CUDA_VISIBLE_DEVICES=$GPU_DEVICE nohup python3 ./eval/run_llm_lp_eval.py \
    --data_flag pos \
    --prompt_mode ${prompt_mode} \
    --data_name ${data_name} \
    --graph_tower clip_gt_${data_name} \
    --graph_data_path ./clip/graph_data_all.pt \
    --end_id ${end_id} > ./logs/${prompt_mode}/${data_name}/eval_${date_time}.log 2>&1 &


# bash ./scripts/run_llm_lp_eval.sh

