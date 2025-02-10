import sys 
sys.path.append('../')
import os
import argparse
import json
import copy
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from predictor.train.conversation import conv_templates, SeparatorStyle
from predictor.train.utils import disable_torch_init
from predictor.model import *
from predictor.model.utils import KeywordsStoppingCriteria
from torch_geometric.data import Data

from predictor.eval.cal_qa_results import eval_result
from sklearn.metrics import accuracy_score
import pathlib
import time


DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


def load_graph(instruct_item, graph_data_path): 
    graph_data_all = torch.load(graph_data_path)
    graph_dict = instruct_item['graph']
    graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
    graph_node_list = copy.deepcopy(graph_dict['node_list'])
    target_node = copy.deepcopy(graph_dict['node_idx'])
    graph_type = copy.deepcopy(instruct_item['id']).split('_')[0]
    graph_node_rep = graph_data_all[graph_type].x[graph_node_list] ## 
    
    cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size
    graph_ret = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))

    return {
        'graph_data': graph_ret, 
        'graph_token_len': cur_token_len
    }


def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_path)
    prompt_file = prompt_file[args.start_id:args.end_id]
    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []
    split_list = list(range(args.start_id, args.end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))
    if len(split_list) == num_gpus: 
        split_list.append(args.end_id)
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1: 
        split_list[-1] = args.end_id
        idx_list[-1] = len(prompt_file)
    else: 
        raise ValueError('error in the number of list')

    if os.path.exists(args.output_path) is False: 
        os.mkdir(args.output_path)
    
    ans_jsons = []
    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]
        
        start_split = split_list[idx]
        end_split = split_list[idx + 1]

        try:
            ans_json = eval_model(args, prompt_file[start_idx:end_idx], start_split, end_split)
            ans_jsons.extend(ans_json)
            print(f"step: {args.step} success")
        except:
            print(f"step: {args.step} fail")
    # with open(args.output_path, "w") as ans_file:
    #     json.dump(ans_jsons,ans_file)


@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx):
    # Model
    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading')

    print('start loading')  
    model = GraphLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    use_graph_start_end = getattr(model.config, "use_graph_start_end", False) # True
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

    graph_tower = model.get_model().graph_tower
    
    # TODO: add graph tower
    # if graph_tower.device.type == 'meta':
    #     print('meta')
    clip_graph, args_graph= load_model_pretrained(CLIP, args.pretrain_graph_model_path)
    graph_tower = graph_transformer(args_graph)
    graph_tower = transfer_param_tograph(clip_graph, graph_tower)
    
    model.get_model().graph_tower = graph_tower.cuda()
    # print(next(graph_tower.parameters()).dtype)
    graph_tower.to(device='cuda', dtype=torch.float16)
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = use_graph_start_end
    if use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
    # TODO: add graph token len
    res_data = []
    all_true, all_pred = [], []
    print(f'total: {len(prompt_file)}')
    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        graph_dict = load_graph(instruct_item, args.graph_data_path)
        graph_token_len = graph_dict['graph_token_len']
        graph_data = graph_dict['graph_data']
        graph_data.graph_node = graph_data.graph_node.to(torch.float16)

        qs = instruct_item["conversations"][0]["value"]
        gound_label = instruct_item["conversations"][1]["value"]
        gound_label = 1 if gound_label== "True" else 0
        all_true.append(gound_label)

        replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
        replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
        qs = qs.replace(DEFAULT_GRAPH_TOKEN, replace_token)

        # if "v1" in args.model_name.lower():
        #     conv_mode = "graphchat_v1"
        # else: 
        #     raise ValueError('Don\'t support this model')
        conv_mode = "graphchat_v1"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                graph_data=graph_data.cuda(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        pred_label = 1 if outputs== "True" else 0
        all_pred.append(pred_label)

        ground_truth = instruct_item["conversations"][1]["value"].split("\n")
        res_data.append({"id": instruct_item["id"], "graph": instruct_item["graph"], "ground_truth":ground_truth, "prediction": outputs}.copy())
        
        prediction_path = os.path.join(args.output_path, '{}_test_{}_{}_{}.json'.format(args.data_flag, start_idx, end_idx, args.step))
        with open(prediction_path, "w") as fout:
            json.dump(res_data, fout, indent=4)
        
    res_dict = eval_result(prediction_path)
    result = f"step: {args.step}, data_flag: {args.data_flag}, " + json.dumps(res_dict) 
    with open(os.path.join(args.output_path, f'{args.data_flag}_results.txt'), "a+") as f:
        f.write(result)

    return res_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="1000")
    parser.add_argument("--data_flag", type=str, default="pos")
    parser.add_argument("--data_name", type=str, default="FB15k-237N")
    parser.add_argument("--prompt_mode", type=str, default="data_llm_graph") 
    parser.add_argument("--graph_tower", type=str, default="clip_gt_FB15k-237N")

    parser.add_argument("--model_name", type=str, default="./checkpoints")
    parser.add_argument("--pretrain_graph_model_path", type=str, default="./clip")
    parser.add_argument("--graph_data_path", type=str, default="./clip/")
    parser.add_argument("--prompting_path", type=str, default=".")
    parser.add_argument("--output_path", type=str, default="./outputs")

    # parser.add_argument("--model_name", type=str, default="./predictor/checkpoints")
    # parser.add_argument("--pretrain_graph_model_path", type=str, default="./predictor/clip")
    # parser.add_argument("--prompting_path", type=str, default="./predictor")
    # parser.add_argument("--output_path", type=str, default="./predictor/outputs")
    # parser.add_argument("--graph_data_path", type=str, default="./predictor/clip/graph_data_all.pt")

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=100)

    args = parser.parse_args()

    args.model_name = os.path.join(args.model_name, args.prompt_mode, args.data_name)
    args.pretrain_graph_model_path = os.path.join(args.pretrain_graph_model_path, args.graph_tower)
    args.prompting_path = os.path.join(args.prompting_path, args.prompt_mode, args.data_name)
    args.prompting_path = os.path.join(args.prompting_path, "test.json")
    args.output_path = os.path.join(args.output_path, args.prompt_mode, args.data_name)

    args.model_name = os.path.join(args.model_name, f"checkpoint-{args.step}")
    run_eval(args, args.num_gpus)

