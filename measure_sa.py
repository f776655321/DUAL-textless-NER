# +
import torch
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
import os 
import json 
import heapq
import pickle 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from torch.nn import LogSoftmax

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='./evaluate-results-sa', type=str)
parser.add_argument('--save_dir', default='./evaluate-store/sa', type=str)
parser.add_argument('--dist_dir', default='./evaluate-distribution-sa', type=str)
# parser.add_argument('--output_fname', default='result', type=str)
parser.add_argument('--mode', default='validation', type=str)
parser.add_argument('--threshold', default=-8, type=float)

parser.add_argument('--range', default="all", type=str)
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
if not os.path.exists(args.dist_dir):
    os.makedirs(args.dist_dir)

def softmax_stable(x):

    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x



action_list = ["positive", "neutral", "negative"]			
answer_list = ["positive", "neutral", "negative"]		
action_prob_distribution = {action:[] for action in action_list}

# action_threshold = None
action_threshold = {action: args.threshold for action in action_list}
# action_threshold["answer_general"] = 16
# action_threshold["statement_close"] = 16
# action_threshold["answer_general"] = 18
# action_threshold["answer_dis"] = -4
# # action_threshold["answer_general"] = 6
# action_threshold["backchannel"] = -6
# action_threshold["statement_instruct"] = -8
# action_threshold["self"] = -10
# action_threshold["question_repeat"] = -7
# action_threshold["apology"] = -7
# action_threshold["answer_agree"] = -8
# # action_threshold["statement_general"] = -5
# action_threshold["disfluency"] = -7
# action_threshold["other"] = -10


prediction_results = defaultdict(set)
prediction_confidence = defaultdict(dict)
ground_truth = defaultdict(set)
batch_size = 32

threshold = args.threshold
    

acc = []
cnt = 0
all_context_id = set()
predictions = None
all_pred = []
all_gt = []
with open(os.path.join(args.save_dir, f"{args.mode}.pickle"), "rb") as f: 
    predictions = pickle.load(f)
    for key in predictions.keys():
        all_context_id.add(key)
      
for context_id in all_context_id: 
    label = predictions[context_id]["label"]
    cls_position = predictions[context_id]['cls_position']
    # all_confidence = []
    # max_confidence = -100000
    start_prob = predictions[context_id]["start_prob"]
    end_prob = predictions[context_id]["end_prob"]
    
    start_prob_region = [start_prob[cls_position[idx] : cls_position[idx + 1]] if idx < len(cls_position) - 1 else start_prob[cls_position[idx]:] for idx in range(len(cls_position))]
    end_prob_region = [end_prob[cls_position[idx] : cls_position[idx + 1]] if idx < len(cls_position) - 1 else end_prob[cls_position[idx]:] for idx in range(len(cls_position))]
    # print(start_prob_region)
    # print(end_prob_region)
    # print("=" * 20)
    # start_prob_region = [prob[0] for prob in start_prob_region]
    # end_prob_region = [prob[-1] for prob in end_prob_region]
    start_prob_region = [max(prob) for prob in start_prob_region]
    end_prob_region = [max(prob) for prob in end_prob_region]
    
    # print(start_prob_region)
    # print(end_prob_region)
    confidence = np.array([start + end for start, end in zip(start_prob_region, end_prob_region)])
    # confidence = softmax_stable(confidence)
    print(confidence) 
    answer = np.argmax(confidence)
    answer = action_list[answer]
    # answer = answer_list[answer]
    # print(answer, label)
    prediction_results[context_id].add(answer)
    for l in label:
        ground_truth[context_id].add(l)
    print(prediction_results[context_id])
    print(ground_truth[context_id])
    print("=" * 20)
    all_pred.append(sorted(list(prediction_results[context_id])))
    all_gt.append(sorted(list(ground_truth[context_id])))
    # print(max(start_prob))
    # print(start_prob[cls_position[0] - 1: cls_position[-1] + 1])
    # print(end_prob[cls_position[0] - 1: cls_position[-1] + 1])

recall = []
precision = []
f1 = []
with open(os.path.join(args.output_dir, f"{args.mode}-{args.range}-prediction.json"), "w") as f:
    combined_result = {}
    prediction_results_with_confidence = {}
    for id in ground_truth.keys():
        ground_truth[id] = list(ground_truth[id])
        prediction_results[id] = list(prediction_results[id]) 
        ground_truth[id] = sorted([gt for gt in ground_truth[id]])
        prediction_results[id] = sorted([pred for pred in prediction_results[id]])

        # prediction_results_with_confidence[id] = sorted([(pred, round(prediction_confidence[id][pred] ,2)) for idx, pred in enumerate(prediction_results[id])], key = lambda x: -x[1])
        
        # TOP-3
        # if len(prediction_results_with_confidence[id]) > 3:
        #     prediction_results_with_confidence[id] = prediction_results_with_confidence[id][:3]
        #     prediction_results[id] = [pred_with_confidence[0] for pred_with_confidence in prediction_results_with_confidence[id][:3]]
        #     print("123")


        combined_result[id] = {"ground_truth": ground_truth[id],
                                "prediction_results": prediction_results[id]}
    json.dump(combined_result, f, indent = 4)

correct_time_by_class = defaultdict(int)
pred_time_by_class = defaultdict(int)
gt_time_by_class = defaultdict(int)

def find_overlap(a, b):
    return len([element for element in a if element in b])

for id, gt in ground_truth.items():
    pred = prediction_results[id]
    overlap = find_overlap(gt, pred)
    
    for g in gt:
        gt_time_by_class[g] += 1
        if g in pred:
            correct_time_by_class[g] += 1
    for p in pred: 
        pred_time_by_class[p] += 1

    # there must be at least one class in ground-truth 
    r = overlap / len(gt)
    
    p = None
    if len(pred) == 0:
        p = 0
    else:
        p = overlap / len(pred)

    f = None
    if p == 0 or r == 0 :
        f = 0 
    else:
        f = 2 * p * r / (p + r)
    recall.append(r)
    precision.append(p)
    f1.append(f) 

macro_f1 = []
for cls in gt_time_by_class:
    try:
        r = correct_time_by_class[cls] / gt_time_by_class[cls] 
        p = correct_time_by_class[cls] / pred_time_by_class[cls]
        f = 2 * p * r / (p + r)
        macro_f1.append(f)
    except:
        print(cls)


total_recall = sum(recall) / len(recall)
total_precision = sum(precision) / len(precision)
print("recall   : ", sum(recall) / len(recall))
print("precision: ", sum(precision) / len(precision))
print("f1       : ", 2 * total_precision * total_recall / (total_precision + total_recall))
print("f1-avg   : ", sum(f1) / len(f1))
print("macro-f1 : ", sum(macro_f1) / len(action_list))

with open(os.path.join(args.output_dir, f"{args.mode}-{args.range}-metrics.txt"), "w") as f:
    for action, threshold in action_threshold.items():
        print(f"{action:15s} {threshold}", file = f)
    print(file = f)
    for cls in gt_time_by_class.keys():
        pred_time = pred_time_by_class[cls]
        gt_time = gt_time_by_class[cls]
        correct_time = correct_time_by_class[cls]
        print(f"{cls}", file = f)
        print(f"prediction count  : {pred_time}", file = f)
        print(f"ground truth count: {gt_time}", file = f)
        print(f"correct count     : {correct_time}\n", file = f)

    print(file = f)
    print("Total", file = f)
    print("recall   : ", sum(recall) / len(recall), file = f)
    print("precision: ", sum(precision) / len(precision), file = f)
    print("f1       : ", 2 * total_precision * total_recall / (total_precision + total_recall), file = f)
    print("f1-avg   : ", sum(f1) / len(f1), file = f)
    print("macro-f1 : ", sum(macro_f1) / len(action_list), file = f)

m = MultiLabelBinarizer().fit(all_gt)
all_gt = m.transform(all_gt)
all_pred = m.transform(all_pred)
json_dict = {}
json_dict["macro"] = {
    "precision": precision_score(all_gt, all_pred, average="macro") * 100,
    "recall": recall_score(all_gt, all_pred, average="macro") * 100,
    "f1": f1_score(all_gt, all_pred, average="macro") * 100,
}
json_dict["micro"] = {
    "precision": precision_score(all_gt, all_pred, average="weighted") * 100,
    "recall": recall_score(all_gt, all_pred, average="weighted") * 100,
    "f1": f1_score(all_gt, all_pred, average="weighted") * 100,
}
json_dict["per_classes"] = {
    action_list[idx]: score
    for idx, score in enumerate(f1_score(all_gt, all_pred, average=None) * 100)
}

with open(os.path.join(args.output_dir, f"{args.mode}-{args.range}-metrics-sk.json"), "w") as fp:
    json.dump(json_dict, fp, sort_keys=True, indent=4)
# # Data for the histogram
# for cls, dist in action_prob_distribution.items():  
#     dist = np.array(dist)
#     # Setting a custom color palette
#     colors = sns.color_palette('husl', 8)

#     # Creating a figure and axes object
#     fig, ax = plt.subplots()

#     # Plotting the histogram
#     ax.hist(dist, bins=10, color=colors[4], edgecolor='black')

#     # Customizing the histogram
#     ax.set_title(cls, fontsize=18, fontweight='bold')
#     ax.set_xlabel('Value Range', fontsize=14)
#     ax.set_ylabel('Frequency', fontsize=14)

#     fig.savefig(os.path.join(args.dist_dir, f"{cls}.jpg"))
