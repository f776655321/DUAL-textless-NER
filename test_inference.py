import torch
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os 
import json 

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from torch.nn import LogSoftmax
from utils import aggregate_dev_result, AOS_scores, Frame_F1_scores

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/work/f776655321/DUAL-textless-NER/new-data', type=str)
parser.add_argument('--model_path', default='/work/f776655321/DUAL-textless-NER/models_5_128_1e-4_warm_500_fp16_2:1_with_n_without_r/checkpoint-1500/', type=str)
parser.add_argument('--output_dir', default='./evaluate-dev', type=str)
parser.add_argument('--output_fname', default='result', type=str)
args = parser.parse_args()
def _get_best_indexes(probs, context_offset, n_best_size):
    # use torch for faster inference
    # do not need to consider indexes for question
    best_indexes = torch.topk(probs[context_offset - 1:],n_best_size).indices
    best_indexes += context_offset - 1
    print(best_indexes)
    return best_indexes
    

def post_process_prediction(start_prob, end_prob,context_offset, n_best_size=10, max_answer_length=500, weight=0.6):
    prelim_predictions = []
    start_prob = start_prob.squeeze()
    end_prob = end_prob.squeeze()
    start_indexes = _get_best_indexes(start_prob,context_offset, n_best_size)
    end_indexes = _get_best_indexes(end_prob,context_offset, n_best_size)
    # if we could have irrelevant answers, get the min score of irrelevant

    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions. This is taken care in _get_best_indexes
            # if start_index >= len(input_id):
            #     continue
            # if end_index >= len(input_id):
            #     continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue

            predict = {
                        'start_prob': start_prob[start_index],
                        'end_prob': end_prob[end_index],
                        'start_idx': start_index,
                        'end_idx': end_index, 
                      }

            prelim_predictions.append(predict)

    prelim_predictions = sorted(prelim_predictions, 
                                key=lambda x: ((1-weight)*x['start_prob'] + weight*x['end_prob']), 
                                reverse=True)
    if len(prelim_predictions) > 0:
        final_start_idx = prelim_predictions[0]['start_idx']
        final_end_idx = prelim_predictions[0]['end_idx']
    else:
        final_start_idx = torch.argmax(start_prob).cpu()
        final_end_idx = torch.argmax(end_prob).cpu()
    return final_start_idx, final_end_idx

def idx2sec(pred_start_idx, pred_end_idx, context_begin, context_cnt):

    context_cnt = context_cnt.squeeze()
    if(pred_start_idx - context_begin != -1):
        start_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:pred_start_idx - context_begin].size()), context_cnt[:pred_start_idx - context_begin])
    else:
        start_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:0].size()), context_cnt[:0])
    
    if(pred_end_idx - context_begin != -1):
        end_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:pred_end_idx - context_begin].size()), context_cnt[:pred_end_idx - context_begin])
    else:
        end_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:0].size()), context_cnt[:0])

    start_idx, end_idx = torch.sum(start_frame_idx), torch.sum(end_frame_idx)

    return float(start_idx*0.02), float(end_idx*0.02)

data_dir = args.data_dir
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
model = LongformerForQuestionAnswering.from_pretrained(args.model_path).cuda()
model.eval()
 
code_dir = os.path.join(data_dir, 'question-code/')
code_passage_dir = os.path.join(data_dir, 'code/dev')

context_id = "context-32_23"

label = "PLACE"

context = np.loadtxt(os.path.join(code_passage_dir, context_id+'.code')).astype(int)
question = np.loadtxt(os.path.join(code_dir, label +'.code')).astype(int)
context_cnt = np.loadtxt(os.path.join(code_passage_dir, context_id+'.cnt')).astype(int)

context += 5
question += 5

'''
<s> question</s></s> context</s>
---------------------------------
<s>: 0
</s>: 2

'''
tot_len = len(question) + len(context) + 4 
code_pair = [[0]+list(question)+[2]+[2]+list(context)+[2]]

context_begin = len(question) + 3

mask = [torch.ones(len(code_pair))]

outputs = model(input_ids=torch.tensor(code_pair).cuda(),
                                      attention_mask=torch.tensor(mask).cuda())

pred_start = torch.argmax(outputs.start_logits, dim=1)
pred_end = torch.argmax(outputs.end_logits, dim=1)

start_prob = softmax(outputs.start_logits, dim=1)
end_prob = softmax(outputs.end_logits, dim=1)

logsoftmax = LogSoftmax(dim=1)
start_logprob = logsoftmax(outputs.start_logits)
end_logprob = logsoftmax(outputs.end_logits)

final_starts, final_ends = [], [] 
print(context_begin)
final_starts, final_ends = post_process_prediction(start_logprob, end_logprob, 
                                                context_begin, 3, 275)


final_start_sec, final_end_sec = idx2sec(final_starts, final_ends, context_begin, torch.tensor(context_cnt).cpu())



print(final_start_sec)
print(final_end_sec)

 









