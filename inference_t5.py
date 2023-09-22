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

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from torch.nn import LogSoftmax
from qa.utils_qa import postprocess_qa_predictions
from qa.t5qa import T5ForQuestionAnswering, T5Model
from qa.enc_t5 import EncT5ForQuestionAnswering, EncLongT5ForQuestionAnswering
from qa.enc_led import EncLEDForQuestionAnswering
from transformers import AutoConfig, AutoTokenizer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='code-data', type=str)
parser.add_argument('--model_path', default='QADAC-t5/checkpoint-1000', type=str)
parser.add_argument('--save_dir', default='./evaluate-store/t5', type=str)
parser.add_argument('--output_dir', default='./evaluate-results', type=str)
parser.add_argument('--dist_dir', default='./evaluate-distribution', type=str)
# parser.add_argument('--output_fname', default='result', type=str)
parser.add_argument('--mode', default='validation', type=str)
args = parser.parse_args()

data_dir = args.data_dir
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
if not os.path.exists(args.dist_dir):
    os.makedirs(args.dist_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


config = AutoConfig.from_pretrained(
    args.model_path,
)
config.num_labels = 2

tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
)

model = EncT5ForQuestionAnswering.from_pretrained(
    args.model_path,
    from_tf=bool(".ckpt" in args.model_path),
    config=config,
).to("cuda")

model.eval()

action_list = ["question_check", "question_repeat", "question_general", "answer_agree", "answer_dis", "answer_general", "apology", "thanks", "acknowledge", "statement_open", "statement_close", "statement_problem", "statement_instruct", "statement_general", "backchannel", "disfluency", "self", "other"]
action_prob_distribution = {action:[] for action in action_list}

class SQADevDataset(Dataset):
    def __init__(self, data_dir):
        if args.mode == "train":
            df = pd.read_csv(os.path.join(data_dir, args.mode + '_code_ans_sampling_positive_5.csv'))

        else: 
            df = pd.read_csv(os.path.join(data_dir, args.mode + '_code_ans.csv'))

        
        code_dir = os.path.join(data_dir, 'question-code-DAC')
        code_passage_dir = os.path.join(data_dir, 'code', args.mode)
        ground_truth = df['label'].values
        context_ids = df['context_id'].values
        
        global action_list
        action_code = {}
        action_cnt = {}
        cumulative_cnt = []
        root_dir = "code-data/question-code-DAC"
        for idx, action in enumerate(action_list):
            code = np.loadtxt(os.path.join(root_dir, action + '.code')).astype(int)
            cnt = len(code)

            action_code[action] = code 
            action_cnt[action] = cnt
            if idx == 0:
                cumulative_cnt.append(cnt)
            else: 
                cumulative_cnt.append(cnt + cumulative_cnt[idx - 1])

        candidate_code = []
        for action in action_list:
            code = action_code[action]
            if code.shape == ():
                print(code)
                # exit()
                code = np.expand_dims(context, axis=-1)
            candidate_code.extend(code)
        candidate_code = [c + 3 for c in candidate_code]
   
        self.encodings = []
        for context_id, gt in tqdm(zip(context_ids, ground_truth)):
            context = np.loadtxt(os.path.join(code_passage_dir, context_id + '.code')).astype(int)
            question = np.loadtxt(os.path.join(code_dir, "question" + '.code')).astype(int)
            context += 3
            question += 3

            if context.shape == ():
                print(context)
                # exit()
                context = np.expand_dims(context, axis=-1)
            '''
            <s> question</s></s> context</s>
            ---------------------------------
            <s>: 0
            </s>: 2
            '''

            #You should modify here to change your input form
            # try:
            tot_len = len(question) + len(context) + len(candidate_code) + 2
    
            # tot_len = len(question) + len(context) + len(candidate_code) + 3
            # except:
            #     tot_len = len(question) + 1 + 4
            #     context = np.array([context])

            if tot_len > 1024 :
                print('length longer than 4096 skip', tot_len)
                # code_pair = [0]+list(question) + [2] + list(context)
                # code_pair = list(question)+[1] + list(context)
                # code_pair = code_pair[:1023] + [1]
                continue
            else:
                code_pair = list(question) + [1] + list(context) + list(candidate_code) + [1]
            
            # cls_position = [cumulative_cnt[idx - 1] + len(question) + len(context) + 2 if idx > 0 else  len(question) + len(context) + 2 for idx in range(len(cumulative_cnt))]
            cls_position = [cumulative_cnt[idx - 1] + len(question) + len(context) + 1 if idx > 0 else  len(question) + len(context) + 1 for idx in range(len(cumulative_cnt))]
            encoding = {}
            # global_attention_mask_len = len(question) + 2
            encoding.update({
                            'context_id': context_id,
                            'input_ids': torch.LongTensor(code_pair), 
                            # 'context_begin': len(question) + 2,  # [0] [2] 
                            'context_begin': len(question) + 1,  # [0] [2] [2]
                            # "global_attention_mask": torch.cat((torch.ones((global_attention_mask_len)), torch.zeros(tot_len - (global_attention_mask_len))), 0),
                            'cls_position': cls_position,
                            'label': gt,
                            'context_len': len(context)
                            })
            self.encodings.append(encoding)

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

    input_ids = pad_sequence([example['input_ids'] for example in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([torch.ones(len(example['input_ids'])) for example in batch], batch_first=True, padding_value=0)
    context_begin = torch.stack([torch.tensor(example['context_begin'], dtype=torch.long) for example in batch])
    context_id = [example['context_id'] for example in batch]
    label = [example['label']for example in batch ]
    context_len = [example['context_len'] for example in batch]
    cls_position = [example['cls_position'] for example in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask, 
        'context_begin': context_begin, 
        # 'context_cnt': context_cnt,
        'cls_position': cls_position,
        'context_id': context_id,
        'label': label,
        'context_len': context_len,
        #  "global_attention_mask": global_attention_mask,
    }

all_prediction = defaultdict(dict)
prediction_results = defaultdict(set)
prediction_confidence = defaultdict(dict)
ground_truth = defaultdict(set)
batch_size = 32
valid_dataset = SQADevDataset(data_dir)
dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_dev_fn, shuffle=False)

if args.mode == "train":
    # df = pd.read_csv(os.path.join(data_dir, args.mode + '_code_ans_sampling_negative_slue_2.csv'))
    df = pd.read_csv(os.path.join(data_dir, args.mode + '_code_ans_sampling_positive_5.csv'))
else:
    df = pd.read_csv(os.path.join(data_dir, args.mode + '_code_ans.csv'))
    # df = pd.read_csv(os.path.join(data_dir, args.mode + '_code_ans_slue.csv'))

with torch.no_grad():
    acc = []
    cnt = 0
    for batch in tqdm(dataloader):
        max_confidence = -1
        prediction = ""
        outputs = model(input_ids=batch['input_ids'].cuda(),
                                    attention_mask=batch['attention_mask'].cuda(),)
                                    #   global_attention_mask=batch['global_attention_mask'].cuda())
        # start_logits: (B, seq_len)
        logsoftmax = LogSoftmax(dim=1)
        # start_logprob = outputs.start_logits
        # end_logprob = outputs.end_logits
        start_logprob = logsoftmax(outputs.start_logits)
        end_logprob = logsoftmax(outputs.end_logits)

        all_confidence = []
        max_confidence = -100000
        best_prediction = -1
        for i in range(start_logprob.shape[0]):
            context_id = batch['context_id'][i]
            context_begin = batch['context_begin'][i]
            context_len = batch['context_len'][i]
            gt = batch['label'][i]
            cls_position = batch['cls_position'][i]
            # print(len(start_logprob[i]))
            # print(cls_position)
            # print(start_logprob[i])
            # print(start_logprob[i][cls_position[0]:cls_position[-1] + 10])
            # print(end_logprob[i][cls_position[0]:cls_position[-1] + 10])
            all_prediction[context_id]["context_begin"] = batch['context_begin'][i]
            all_prediction[context_id]["context_len"] = batch['context_len'][i]
            all_prediction[context_id]["start_prob"] = start_logprob[i].detach().cpu().tolist()
            all_prediction[context_id]["end_prob"] = end_logprob[i].detach().cpu().tolist()
            all_prediction[context_id]["cls_position"] = cls_position
            if "label" in all_prediction[context_id].keys():
                all_prediction[context_id]["label"].add(gt)
            else: 
                all_prediction[context_id]["label"] = set()
                all_prediction[context_id]["label"].add(gt)

with open(os.path.join(args.save_dir, f"{args.mode}.pickle"), "wb") as f:
        pickle.dump(all_prediction, f)

        
