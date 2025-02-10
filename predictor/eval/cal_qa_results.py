import argparse
import glob
import json
import os
import re
import string
from sklearn.metrics import precision_score

def normalize(s):
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

def eval_exact_match(prediction, answer):
    matched = 0.
    for a in answer:
        if prediction == a:
            matched += 1
    return matched / len(answer)

def match(s1, s2):
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)

def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0

def eval_mrr(prediction, answer):
    mrr = 0
    for a in answer:
        for index, p in enumerate(prediction):
            p = normalize(p)
            a = normalize(a)
            if a in p:
                mrr += 1/(index+1)
    mrr = mrr / len(answer)
    return mrr

def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)

    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def eval_result(predict_file, topk = -1):
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    em_list = []
    mrr_list = []
    # eval_data_list = []

    with open(predict_file, 'r') as f:
        predict_data_list = json.load(f)

    for data in predict_data_list:
        id = data['id']
        prediction = data['prediction']
        answer = data['ground_truth']

        if not isinstance(prediction, list):
            prediction = prediction.split("\n")
        else:
            prediction = extract_topk_prediction(prediction, topk)
        f1_score, precision_score, recall_score = eval_f1(prediction, answer)
        f1_list.append(f1_score)
        precission_list.append(precision_score)
        recall_list.append(recall_score)
        prediction_str = ' '.join(prediction)
        acc = eval_acc(prediction_str, answer)
        hit = eval_hit(prediction_str, answer)
        acc_list.append(acc)
        hit_list.append(hit)
        em = eval_exact_match(prediction_str, answer)
        em_list.append(em)
        mrr = eval_mrr(prediction, answer)
        mrr_list.append(mrr)

        # eval_data_list.append({'id': id, 'prediction': prediction, 'ground_truth': answer, 
        #                        'acc': acc, 'hit': hit, 'f1': f1_score, 'precission': precision_score, 'recall': recall_score})
    
    acc = sum(acc_list) * 100 / len(acc_list)
    hit = sum(hit_list) * 100 / len(hit_list)
    f1 = sum(f1_list) * 100 / len(f1_list)
    pre = sum(precission_list) * 100 / len(precission_list)
    rec = sum(recall_list) * 100 / len(recall_list)
    em = sum(em_list) * 100 / len(em_list)
    mrr = sum(mrr_list) * 100 / len(mrr_list)

    result_dict = {"EM": em, "Accuracy": acc, "Hit": hit, "F1": f1, "Precision": pre, "Recall": rec, "MRR": mrr}
    # result_str = " Accuracy: "+str(acc) + " Hit: "+str(hit) + " F1: "+str(f1) + " Precision: "+str(pre) + " Recall: "+str(rec)  + " MRR: "+str(mrr)
    result_str = json.dumps(result_dict)
    print(result_str)    
    return result_dict


def eval_all(args):
    print("eval_all")
    data_path = os.path.join(args.root_path, args.data_name)
    filenames=os.listdir(data_path)
    filename_dict = {}
    for filename in filenames:
        if "pos" in filename and ".json" in filename:
            step = filename.split("_")[-1]
            step = int(step.split(".")[0])
            filename_dict[step] = filename
    filename_dict = dict(sorted(filename_dict.items(), key=lambda item: item[0]))
    
    max_hit = 0
    for step, filename in filename_dict.items():
        if "pos" in filename and ".json" in filename:

            predict_file = os.path.join(data_path, filename)
            res_dict = eval_result(predict_file)
            res_str = ""
            for metric, value in res_dict.items():
                res_str += metric + ": " + str(value) + " "
        print(f"step: {str(step)}", res_str)

        # max_hit = max(max_hit,res_dict["Hit"])
        # print(f"Histroy Max Hit: {max_hit}" )
         

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--cal_f1', type=bool, default=True)
    argparser.add_argument('--top_k', type=int, default=-1)
    argparser.add_argument('--root_path', type=str, default="./outputs")
    argparser.add_argument('--data_name', type=str, default='data_qa_kgc_llm_graph_new/FB15k-237')
    argparser.add_argument('--predict_file', type=str, default='pos_test_0_8226_65000.json')
    args = argparser.parse_args()

    args.predict_file = os.path.join(args.root_path, args.data_name, args.predict_file)
    # eval_result(args.predict_file)

    eval_all(args)
