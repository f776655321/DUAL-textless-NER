# +
import torch
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os 
import json 
import heapq

from pandas import Interval
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from torch.nn import LogSoftmax
from utils import aggregate_dev_result, AOS_scores, Frame_F1_scores

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/work/f776655321/DUAL-textless-NER/code-data', type=str)
parser.add_argument('--model_path', default='/work/f776655321/DUAL-textless-NER/SQA_specaug_unk13000/checkpoint-13000/', type=str)
parser.add_argument('--output_dir', default='./evaluate-dev', type=str)
parser.add_argument('--output_fname', default='result', type=str)
args = parser.parse_args()

data_dir = args.data_dir
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
model = LongformerForQuestionAnswering.from_pretrained(args.model_path).cuda()
model.eval()

# threshold_search_space = [-0.6,-1.1,-1.5,-1.8,-2,-2.5,-2.8,-3.3,-3.8,-4.2,-4.5]
threshold_search_space = [-2.5]
# answer_length_search_space = [20,40,80,120,150,200,250]
answer_length_search_space = [80]
'''
post-processing the answer prediction
'''
mode = 'dev'

k = 10
def _get_best_indexes(probs, context_offset):
    # use torch for faster inference
    # do not need to consider indexes for question
    probs = probs[context_offset:]

    #threshold method
    # mask = probs > threshold
    # best_indexes = torch.nonzero(mask)
    # best_indexes = best_indexes.reshape(-1)
    # best_indexes += context_offset - 1

    #top-k method
    if(k < len(probs)):
        top_values, top_indices = torch.topk(probs, k)
    else:
        top_values, top_indices = torch.topk(probs, len(probs))

    best_indexes = top_indices + context_offset

    return best_indexes

def post_process_prediction(start_prob, end_prob,context_offset,context_id,context_len,threshold,max_answer_length,weight = 0.6):
        
    start_prob = start_prob.squeeze()
    end_prob = end_prob.squeeze()

    start_indexes = _get_best_indexes(start_prob,context_offset)
    end_indexes = _get_best_indexes(end_prob,context_offset)

    original_start = start_indexes.clone()
    original_end = end_indexes.clone()

    final_start_indexes = []
    final_end_indexes = []

    start_len = len(start_indexes)
    end_len = len(end_indexes)

    prob_len = len(start_prob)

    #prevent unlegal output
    if(context_offset - 1 in start_indexes and context_offset - 1 in end_indexes):
        negative_score = start_prob[context_offset + context_len + 1] + end_prob[context_offset + context_len + 1]
    
    else:
        negative_score = -100000000000


    mask = end_indexes != context_offset + context_len + 1
    end_indexes = end_indexes[mask]
    end_len -= 1

    mask = start_indexes != context_offset + context_len + 1
    start_indexes = start_indexes[mask]
    start_len -= 1

    prelim_predictions = []

    for start_index in start_indexes:
        for end_index in end_indexes:
            
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
                        'score': start_prob[start_index] + end_prob[end_index]
                      }

            prelim_predictions.append(predict)

    prelim_predictions = sorted(prelim_predictions, 
                                key=lambda x: x['score'],
                                reverse=True)
    
    for candidate in prelim_predictions:
        if(candidate['score'] >= threshold and candidate['score'] >= negative_score):
            final_start_indexes.append(candidate['start_idx'].item())
            final_end_indexes.append(candidate['end_idx'].item())
    
    if(len(final_start_indexes) == 0):
        final_start_indexes.append((context_offset + context_len + 1).item())
        final_end_indexes.append((context_offset + context_len + 1).item())
    
    # print(final_start_indexes)
    # print(final_end_indexes)
    # input()

    output_start = []
    output_end = []

    for start,end in zip(final_start_indexes,final_end_indexes):
        candidate_pair = Interval(start,end,closed='both')

        overlapping = False

        for start_,end_ in zip(output_start,output_end):
            decide_pair = Interval(start_,end_,closed='both')

            if decide_pair.overlaps(candidate_pair):
                overlapping = True
                break
        
        if overlapping == False:
            output_start.append(start)
            output_end.append(end)
    
    # print(output_start)
    # print(output_end)
    # input()
    
    # print(start_prob[context_offset - 1])
    # print(end_prob[context_offset - 1])
    # input()
    # candi_pobs = []

    # if start_len > end_len:
    #     for i in range(start_len):
    #         candi_pobs.append( ( start_prob[start_indexes[i]],start_indexes[i] ) )

    #     n_start = heapq.nlargest(end_len,candi_pobs)
    #     start_indexes = []

    #     for i in range(end_len):
    #         start_indexes.append(n_start[i][1])

    #     start_indexes.sort()
    # elif start_len < end_len:
    #     for i in range(end_len):
    #         candi_pobs.append( (end_prob[end_indexes[i]],end_indexes[i]) )

    #     n_start = heapq.nlargest(start_len,candi_pobs)
    #     end_indexes = []

    #     for i in range(start_len):
    #         end_indexes.append(n_start[i][1])

    #     end_indexes.sort()
    # n = min(start_len,end_len)

    # if(n == 0):
    #     final_start_idx = torch.argmax(start_prob).cpu()
    #     final_end_idx = torch.argmax(end_prob).cpu()
    #     final_start_indexes.append(final_start_idx)
    #     final_end_indexes.append(final_end_idx)

    # else:
    #     first_index = 0
    #     second_index = 0
    #     count = 0
    #     while first_index < n and second_index < n:
    #         start = start_indexes[first_index]
    #         end = end_indexes[second_index]
    #         if end >= start:
    #             final_start_indexes.append(start_indexes[first_index])
    #             final_end_indexes.append(end_indexes[second_index])
    #             first_index += 1
    #             second_index += 1
    #             break
    #         else:
    #             second_index += 1

    #     while first_index < n and second_index < n:
    #         start = start_indexes[first_index]
    #         end = end_indexes[second_index]
    #         if end >= start:
    #             if start >= final_end_indexes[count]:
    #                 final_start_indexes.append(start)
    #                 final_end_indexes.append(end)
    #                 first_index += 1
    #                 second_index += 1
    #                 count += 1
    #             else:
    #                 first_index += 1
    #         else:
    #             second_index += 1

    #     final_len = len(final_start_indexes)
    #     if final_len > 1 and final_start_indexes[0] == context_offset - 1 and final_end_indexes[0] == context_offset - 1:
    #         negative_start_prob = start_prob[final_start_indexes[0]]
    #         negative_end_prob = end_prob[final_end_indexes[0]]
    #         target_prob = (1 - weight) * negative_start_prob + weight * negative_end_prob
    #         check = False
    #         temp_start_indexs = []
    #         temp_end_indexs = []
    #         for i in range(1,final_len):
    #             # prob = (1 - weight) * start_prob[final_start_indexes[i]] + weight * end_prob[final_end_indexes[i]]
    #             # if prob > target_prob:
    #             #     del final_start_indexes[0]
    #             #     del final_end_indexes[0]
    #             #     check = True
    #             #     break
    #             if start_prob[final_start_indexes[i]] > negative_start_prob and end_prob[final_end_indexes[i]] > negative_end_prob:
    #                 temp_start_indexs.append(final_start_indexes[i])
    #                 temp_end_indexs.append(final_end_indexes[i])
    #                 check = True
                    
    #         if check == False:
    #             final_start_indexes = [ final_start_indexes[0] ]
    #             final_end_indexes = [ final_end_indexes[0] ]
    #         else:
    #             final_start_indexes = temp_start_indexs
    #             final_end_indexes = temp_end_indexs

    # if final_start_indexes == []:
        
    #     final_start_idx = torch.argmax(start_prob).cpu()
    #     final_end_idx = torch.argmax(end_prob).cpu()
    #     final_start_indexes.append(final_start_idx)
    #     final_end_indexes.append(final_end_idx)

    return output_start,output_end

def process_overlapping(start_probs,end_probs,starts,ends,context_begins,weight = 0.6):
    total = []
    i = 0
    for start_array,end_array,context_begin in zip(starts,ends,context_begins):
        for start,end in zip(start_array,end_array):
            total.append((start - context_begin + 1,end - context_begin + 1,i))
        i += 1
        
    total.sort()

    new_starts = [[] for _ in range(len(context_begins))]
    new_ends = [[] for _ in range(len(context_begins))]

    start_index = 0
    n = len(total)
    
    for i in range(1,n):
        label = total[start_index][2]
        start = total[start_index][0] + context_begins[label] - 1
        end = total[start_index][1] + context_begins[label] - 1

        if((total[start_index][0] < total[i][0] and total[i][0] < total[start_index][1]) or (total[start_index][0] < total[i][1] and  total[i][1] < total[start_index][1])):
            prob1 = (1 - weight) * start_probs[label][start] + weight * end_probs[label][end]

            label2 = total[i][2]
            start2 = total[i][0] + context_begins[label2] - 1
            end2 = total[i][1] + context_begins[label2] - 1
            prob2 = (1 - weight) * start_probs[label2][start2] + weight * end_probs[label2][end2]

            if(prob1 < prob2):
                start_index = i
            # if(start_probs[label][start] < start_probs[label2][start2] and end_probs[label][end] < end_probs[label2][end2]):
            #     start_index = i
        else:
            new_starts[label].append(start)
            new_ends[label].append(end)
            start_index = i
        

    label = total[start_index][2]
    start = total[start_index][0] + context_begins[label] - 1
    end = total[start_index][1] + context_begins[label] - 1
    new_starts[label].append(start)
    new_ends[label].append(end)

    n = len(new_starts)
    for i in range(n):
        if(new_starts[i] == []):
            new_starts[i].append(context_begins[i] - 1)
            new_ends[i].append(context_begins[i] - 1)

    return new_starts,new_ends
class SQADevDataset(Dataset):
    def __init__(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, mode + '_frame_inference.csv'))
        
        code_dir = os.path.join(data_dir, 'question-code/')
        code_passage_dir = os.path.join(data_dir, 'code',mode)
        context_ids = df['context_id'].values
        Combined_label = ['PLACE','QUANT','ORG','WHEN','NORP','PERSON','LAW']
        self.encodings = []
        for context_id in tqdm(context_ids):
            context = np.loadtxt(os.path.join(code_passage_dir, context_id+'.code')).astype(int)
            context_cnt = np.loadtxt(os.path.join(code_passage_dir, context_id+'.cnt')).astype(int)
            for label in Combined_label:
                question = np.loadtxt(os.path.join(code_dir, label +'.code')).astype(int)
                # question_cnt = np.loadtxt(os.path.join(code_dir, question_id+'.cnt')).astype(int)
                # 0~4 index is the special token, so start from index 5
                # the size of discrete token is 128, indexing from 5~132
                context += 5
                question += 5

                '''
                <s> question</s></s> context</s>
                ---------------------------------
                <s>: 0
                </s>: 2
                <unk>: 3
                '''
                tot_len = len(question) + len(context) + 4
                
                
                if tot_len > 4096 :
                    print('length longer than 4096: ', tot_len)
                    code_pair = [0]+list(question)+[2]+[2]+list(context)
                    code_pair = code_pair[:4094] + [2] + [3]
                else:
                    code_pair = [0]+list(question)+[2]+[2]+list(context)+ [2] + [3]
                

                encoding = {}

                encoding.update({'input_ids': torch.LongTensor(code_pair), 
                                'context_begin': len(question) + 3,  # [0] [2] [2]
                                'context_cnt': context_cnt,
                                'context_id':context_id,
                                'label':label,
                                'context_len': len(context)
                                })
                self.encodings.append(encoding)
                context -= 5

    def __len__(self):
        return len(self.encodings)
    def __getitem__(self, idx):
        return self.encodings[idx]
        

def collate_dev_fn(batch):
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
        A dictionary of tensors
    """
    # padding
    for example in batch:
        if len(example['input_ids']) > 4096:
            print('too long:', len(example['input_ids']))
    input_ids = pad_sequence([example['input_ids'] for example in batch], batch_first=True, padding_value=1)
    attention_mask = pad_sequence([torch.ones(len(example['input_ids'])) for example in batch], batch_first=True, padding_value=0)
    context_begin = torch.stack([torch.tensor(example['context_begin'], dtype=torch.long) for example in batch])
    context_cnt = pad_sequence([torch.tensor(example['context_cnt']) for example in batch], batch_first=True, padding_value=0)  
    context_id = [ example['context_id']for example in batch ]
    label = [ example['label']for example in batch ]
    context_len = [example['context_len'] for example in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask, 
        'context_begin': context_begin, 
        'context_cnt': context_cnt,
        'context_id':context_id,
        'label':label,
        'context_len': context_len
    }


def idx2sec(pred_start_idx, pred_end_idx, context_begin, context_cnt,context_len):
    context_cnt = context_cnt.squeeze()

    if(pred_start_idx != context_len + context_begin + 1):
        start_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:pred_start_idx - context_begin].size()), context_cnt[:pred_start_idx - context_begin])
    else:
        start_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:0].size()), context_cnt[:0])
    
    if(pred_end_idx != context_len + context_begin + 1):
        end_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:pred_end_idx - context_begin].size()), context_cnt[:pred_end_idx - context_begin])
    else:
        end_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:0].size()), context_cnt[:0])

    start_idx, end_idx = torch.sum(start_frame_idx), torch.sum(end_frame_idx)
    
    # return float(start_idx*0.02), float(end_idx*0.02)
    return start_idx.item(), end_idx.item()

##############

#TODO: read all the document in inference

max_ff1 = -1
max_aos = -1
best_threshold = 0
best_ans_length = 0

agg_f1s_after_sec = []
agg_AOSs = []
count = []

for threshold in threshold_search_space:
    for answer_length in answer_length_search_space:
        batch_size = 7
        valid_dataset = SQADevDataset(data_dir)
        dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_dev_fn, shuffle=False)


        df = pd.read_csv(os.path.join(data_dir, mode + '_frame_inference.csv'))

        start_secs = df['start'].values
        end_secs = df['end'].values
        context_id = df['context_id'].values

        f1s_after_sec = {'PLACE':[],'QUANT':[],'ORG':[],'WHEN':[],'NORP':[],'PERSON':[],'LAW':[]}
        AOSs =  {'PLACE':[],'QUANT':[],'ORG':[],'WHEN':[],'NORP':[],'PERSON':[],'LAW':[]}
        Combined_label = ['PLACE','QUANT','ORG','WHEN','NORP','PERSON','LAW']
        with torch.no_grad():
            total_diff = 0
            i = 0
            for batch in tqdm(dataloader):
                
                outputs = model(input_ids=batch['input_ids'].cuda(),
                                            attention_mask=batch['attention_mask'].cuda())
                # start_logits: (B, seq_len)
                logsoftmax = LogSoftmax(dim=1)
                start_logprob = logsoftmax(outputs.start_logits)
                end_logprob = logsoftmax(outputs.end_logits)
            
                final_starts, final_ends = [], []
                if batch_size == 1:
                    final_starts, final_ends = post_process_prediction(start_logprob, end_logprob, 
                                                                batch['context_begin'], 275)
                else: 
                    for j in range(start_logprob.shape[0]):
                        # final_start and end is an array which contains the pairs of ( start_index,end_index )
                        final_start, final_end = post_process_prediction(start_logprob[j], end_logprob[j], 
                                                                    batch['context_begin'][j],batch['context_id'][j],batch['context_len'][0],threshold,answer_length)
                        final_starts.append(final_start)
                        final_ends.append(final_end)
                # final_starts, final_ends = process_overlapping(start_logprob,end_logprob,final_starts,final_ends,batch['context_begin'])
                final_start_secs, final_end_secs = [], []
                iterate = 0
                for final_start, final_end, context_begin, context_cnt  in zip(final_starts, final_ends, batch['context_begin'].cpu(), batch['context_cnt'].cpu()):
                    final_start_secs.append([])
                    final_end_secs.append([])

                    for start_index,end_index in zip(final_start,final_end):
                        final_start_sec, final_end_sec = idx2sec(start_index, end_index, context_begin, context_cnt,batch['context_len'][0])
                        final_start_secs[iterate].append(final_start_sec)
                        final_end_secs[iterate].append(final_end_sec)
                    iterate += 1

                f1_after_sec,diff = Frame_F1_scores(start_secs[i], end_secs[i],
                                    final_start_secs, final_end_secs,Combined_label)
                total_diff += diff
                AOS_sec = AOS_scores(start_secs[i], end_secs[i],
                                    final_start_secs, final_end_secs,Combined_label)

                for key in f1_after_sec:

                    f1s_after_sec[key] += f1_after_sec[key]
                    AOSs[key] += AOS_sec[key]
                i += 1

        total_f1s = []
        total_AOSs = []

        for label in Combined_label:
            total_f1s += f1s_after_sec[label]

            total_AOSs += AOSs[label]

        agg_dev_Frame_F1_score_after_sec = aggregate_dev_result(total_f1s)
        agg_dev_AOSs = aggregate_dev_result(total_AOSs)

        if(agg_dev_Frame_F1_score_after_sec > max_ff1):
            max_ff1 = agg_dev_Frame_F1_score_after_sec
            max_aos = agg_dev_AOSs

            best_threshold = threshold
            best_ans_length = answer_length

            agg_f1s_after_sec = []
            agg_AOSs = []
            count = []
            
            Combined_label = ['PLACE','QUANT','ORG','WHEN','NORP','PERSON','LAW']

            for label in Combined_label:
                count.append(len(f1s_after_sec[label]))
                agg_f1s_after_sec.append(aggregate_dev_result(f1s_after_sec[label]))
                agg_AOSs.append(aggregate_dev_result(AOSs[label]))


output_df = pd.DataFrame(list(zip(Combined_label, agg_f1s_after_sec, agg_AOSs,count)),
                columns=['label', 'f1', 'AOS','number'])

output_df.to_csv(os.path.join(args.output_dir, args.output_fname+'.csv'),index=False)

        
print(args.output_fname)
print('best F1: ', max_ff1)
print('best aos: ', max_aos)
print('best_threshold: ',best_threshold)
print('best_ans_length: ',best_ans_length)

with open( os.path.join(args.output_dir,args.output_fname+'.txt'), 'w') as f:
    f.write(args.output_fname + '\n')
    f.write('post-processed f1 sec: ' + str(max_ff1)+ '\n')
    f.write('post-processed aos sec: ' + str(max_aos))
