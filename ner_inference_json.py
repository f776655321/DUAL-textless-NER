# +
import torch
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os 
import json 
import heapq

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from torch.nn import LogSoftmax
from utils import post_process_prediction, process_overlapping, find_overlapframe, calculate_FF1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/work/f776655321/DUAL-textless-NER/code-data', type=str)
parser.add_argument('--model_path', default='/work/f776655321/DUAL-textless-NER/SQAPretrain_simple_hubert_nowarp/checkpoint-11000', type=str)
parser.add_argument('--output_dir', default='./output', type=str)
parser.add_argument('--output_fname', default='result', type=str)
args = parser.parse_args()

data_dir = args.data_dir
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

model = LongformerForQuestionAnswering.from_pretrained(args.model_path).cuda()
model.eval()

'''
post-processing the answer prediction
'''
mode = 'dev'

k = 10

class SQADevDataset(Dataset):
    def __init__(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, mode + '_frame_inference.csv'))
        # df = pd.read_csv(os.path.join(data_dir, mode + '_context_id.csv'))
        code_dir = os.path.join(data_dir, 'question-code-c128/')
        code_passage_dir = os.path.join(data_dir, 'code',mode)
        context_ids = df['context_id'].values
        Combined_label = ['PLACE','QUANT','ORG','WHEN','NORP','PERSON','LAW']
        self.encodings = []
        for context_id in tqdm(context_ids):
            print(context_id)
            input()
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
                '''

                #You should modify here to change your input form
                tot_len = len(question) + len(context) + 4
                
                if tot_len > 4096 :
                    print('length longer than 4096: ', tot_len)
                    code_pair = [0]+list(question)+[2]+[2]+list(context)
                    code_pair = code_pair[:4094] + [133,2]
                else:
                    code_pair = [0]+list(question)+[2]+[2]+list(context) + [133,2]
                

                encoding = {}

                encoding.update({'input_ids': torch.LongTensor(code_pair), 
                                'context_begin': len(question) + 3,  # [0] [2] [2]
                                'context_cnt': context_cnt,
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
    label = [ example['label']for example in batch ]
    context_len = [example['context_len'] for example in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask, 
        'context_begin': context_begin, 
        'context_cnt': context_cnt,
        'label':label,
        'context_len': context_len
    }

# negative_start is the index indicate the negative example answer start, and so negative_end
def idx2sec(pred_start_idx, pred_end_idx, context_begin, context_cnt,negative_start,negative_end):
    context_cnt = context_cnt.squeeze()

    if(pred_start_idx != negative_start):
        start_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:pred_start_idx - context_begin].size()), context_cnt[:pred_start_idx - context_begin])
    else:
        start_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:0].size()), context_cnt[:0])
    
    if(pred_end_idx != negative_end):
        end_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:pred_end_idx - context_begin].size()), context_cnt[:pred_end_idx - context_begin])
    else:
        end_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:0].size()), context_cnt[:0])

    start_idx, end_idx = torch.sum(start_frame_idx), torch.sum(end_frame_idx)
    
    # output milliseconds
    return float(start_idx.item()*0.02*1000), float(end_idx.item()*0.02*1000)

Combined_label = ['PLACE','QUANT','ORG','WHEN','NORP','PERSON','LAW']
batch_size = 7
valid_dataset = SQADevDataset(data_dir)
dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_dev_fn, shuffle=False)

original_id = pd.read_csv('/work/f776655321/DUAL-textless-NER/slue-voxpopuli/slue-voxpopuli_' + mode  +'.tsv', delimiter='\t', usecols=[0])
original_id = original_id['id'].tolist()

print(original_id)
input()

output = dict()

thresholds = [-5.5, -5, -3.8, -5.5, -5, -1.5, -1.1]
answer_length = 80

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

        #postprocess the output
        if batch_size == 1:
            final_starts, final_ends = post_process_prediction(start_logprob, end_logprob, 
                                                        batch['context_begin'], 275)
        else: 
            for j in range(start_logprob.shape[0]):
                
                #negative index
                negative_start = batch['context_begin'][j].item() + batch['context_len'][0]
                # negative_start = 0
                negative_end = batch['context_begin'][j].item() + batch['context_len'][0]
                # negative_end = 0

                # final_start and end is an array which contains the pairs of ( start_index,end_index )
                final_start, final_end = post_process_prediction(start_logprob[j], end_logprob[j], 
                                                            batch['context_begin'][j],batch['context_len'][0],negative_start,negative_end,thresholds[j],answer_length)
                final_starts.append(final_start)
                final_ends.append(final_end)

        # transform the output index to frame index
        final_start_secs, final_end_secs = [], []
        iterate = 0
        for final_start, final_end, context_begin, context_cnt  in zip(final_starts, final_ends, batch['context_begin'].cpu(), batch['context_cnt'].cpu()):
            final_start_secs.append([])
            final_end_secs.append([])

            #negative index
            negative_start = context_begin.item() + batch['context_len'][0]
            # negative_start = 0
            negative_end = context_begin.item() + batch['context_len'][0]
            # negative_end = 0

            for start_index,end_index in zip(final_start,final_end):
                final_start_sec, final_end_sec = idx2sec(start_index, end_index, context_begin, context_cnt,negative_start,negative_end)
                final_start_secs[iterate].append(final_start_sec)
                final_end_secs[iterate].append(final_end_sec)

            iterate += 1

        result = [(a[0], b[0]) for a, b in zip(final_start_secs, final_end_secs) if a[0] != 0.0 or b[0] != 0.0]
        
        output[original_id[i]] = result

        i += 1

with open( mode + '.json', 'w') as file:
    json.dump(output, file)
