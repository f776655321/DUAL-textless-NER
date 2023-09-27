# +
import torch
from torch.nn.parameter import Parameter
import nlp
import torch.nn as nn
from transformers import LongformerTokenizerFast
import os 
import numpy as np
import json
import pandas as pd
from tqdm import tqdm 
import argparse
import yaml 
import json
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from datasets import load_metric
metric = load_metric('accuracy')

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     start_prob = predictions[0]
#     end_prob = predictions[1]
#     start_label = labels[0]
#     end_label = labels[1]
#     start_prediction = np.argmax(start_prob, axis=1)
#     end_prediction = np.argmax(end_prob, axis=1)
#     labels = start_label + end_label
#     predictions = start_prediction + end_prediction
#     return metric.compute(predictions=predictions, references=labels)

class SQADataset(Dataset):
    def __init__(self, data_dir, mode='train', idx_offset=5):   
        if(mode == 'train'):
            # df = pd.read_csv(os.path.join(data_dir, mode + '_code_ans.csv'))     
            df = pd.read_csv(os.path.join(data_dir, mode + '_code_ans.csv'))     
        else:
            # df = pd.read_csv(os.path.join(data_dir, mode + '_code_ans.csv'))
            df = pd.read_csv(os.path.join(data_dir, mode + '_code_ans.csv'))
        # TODO
        code_dir = os.path.join(data_dir, 'question-code-DAC')
        # code_dir = os.path.join(data_dir, 'question-code-dac-slue')
        # TODO
        code_passage_dir = os.path.join(data_dir, 'code/' + mode + '/')
        context_id = df['context_id'].values
        code_start = df['code_start'].values
        code_end = df['code_end'].values

        labels = df['label'].values
        self.encodings = []

        action_list = ["question_check", "question_repeat", "question_general", "answer_agree", "answer_dis", "answer_general", "apology", "thanks", "acknowledge", "statement_open", "statement_close", "statement_problem", "statement_instruct", "statement_general", "backchannel", "disfluency", "self", "other"]
        action_code = {}
        candidate_code = []
        root_dir = "code-data/question-code-DAC"
        for idx, action in enumerate(action_list):
            code = np.loadtxt(os.path.join(root_dir, action + '.code')).astype(int)
            action_code[action] = code 
            if code.shape == ():
                print(code)
                code = np.expand_dims(context, axis=-1)
            candidate_code.extend(code)
        candidate_code = [c + idx_offset for c in candidate_code]
        
        question = np.loadtxt(os.path.join(code_dir, "question" + '.code')).astype(int)
        if question.shape == ():
            print(question)
            question = np.expand_dims(question, axis=-1)
        question += idx_offset
        
        for context_id, label, start_idx, end_idx in tqdm(zip(context_id, labels, code_start, code_end), total=len(context_id)):
            context = np.loadtxt(os.path.join(code_passage_dir, context_id+'.code')).astype(int)
            
            if context.shape == ():
                print(context)
                context = np.expand_dims(context, axis=-1)

            # 0~4 index is the special token, so start from index 5
            # the size of discrete token is 128, indexing from 5~132
            context += idx_offset

            '''
            <s> question</s></s> context</s>.
            ---------------------------------
            <s>: 0
            </s>: 2
            '''
            tot_len = len(question) + len(context) + len(candidate_code) + 4
            # tot_len = len(question) + len(context) + len(candidate_code) + 3
            
            if(end_idx > 0):
                start_positions = 1 + len(question) + 1 + 1 + start_idx
                end_positions = 1 + len(question) + 1 + 1 + end_idx
                # start_positions = 1 + len(question)  + 1 + start_idx
                # end_positions = 1 + len(question) + 1 + end_idx
                # print(tot_len, start_positions, end_positions)
            else:
                print(start_idx, end_idx)
                start_positions = 0
                end_positions = 0

            code_pair = [0] + list(question) + [2] + [2] + list(context) + list(candidate_code) + [2]
            # code_pair = [0] + list(question)  + [2] + list(context) + list(candidate_code) + [2]
            # global_attention_mask_len = len(question) + 2
            if tot_len != len(code_pair) :
                print(context_id, end_positions, tot_len)
            encoding = {}

            encoding.update({'input_ids': torch.LongTensor(code_pair),
                                # 'context_begin': len(question) + 3,  # [0] [2] [2]
                                # 'context_cnt': context_cnt, 
                                'start_positions': start_positions,
                                'end_positions': end_positions,
                                # "global_attention_mask": torch.cat((torch.ones((global_attention_mask_len)), torch.zeros(tot_len - (global_attention_mask_len))), 0),
                                # "global_attention_mask_len": global_attention_mask_len,
                                
                                # 'context_len': len(context)
                            })
            self.encodings.append(encoding)
        # exit()
    def __len__(self):
        return len(self.encodings)
    def __getitem__(self, idx):
        return self.encodings[idx]
        

def collate_batch(batch):
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
    # max_len = len(attention_mask[0])
    # global_attention_mask = pad_sequence([example['global_attention_mask'] for example in batch], batch_first=True, padding_value=0) 
    # global_attention_mask[0] = nn.ConstantPad1d((0, max_len - global_attention_mask[0].shape[0]), 0)(global_attention_mask[0])
    # global_attention_mask = pad_sequence([torch.ones((example['global_attention_mask_len'])) for example in batch], batch_first=True, padding_value=0,) 
    
    start_positions = torch.stack([torch.tensor(example['start_positions'], dtype=torch.long) for example in batch])
    end_positions = torch.stack([torch.tensor(example['end_positions'], dtype=torch.long) for example in batch])
    # context_begin = torch.stack([torch.tensor(example['context_begin'], dtype=torch.long) for example in batch])
    # context_cnt = pad_sequence([torch.tensor(example['context_cnt']) for example in batch], batch_first=True, padding_value=0)  
    # context_len = [example['context_len'] for example in batch]
    return {
        'input_ids': input_ids,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'attention_mask': attention_mask, 
        # "global_attention_mask": global_attention_mask,
        # 'context_begin': context_begin, 
        # 'context_cnt': context_cnt,
        # 'label':label,
        # 'context_len': context_len
    }


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn import LogSoftmax

from transformers import LongformerForQuestionAnswering, LongformerTokenizerFast, EvalPrediction
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
    get_linear_schedule_with_warmup,
    AdamW,
)
from accelerate import Accelerator


logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_len: Optional[int] = field(
        default=4096,
        metadata={"help": "Max input length for the source text"},
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # we will load the arguments from a json file,
    json_file=os.path.abspath('args_trainer.json')
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('args_trainer.json'))
 
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    # tokenizer = LongformerTokenizerFast.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    # )
    model = LongformerForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    for name in model.state_dict():
        print(name)
    # print(model)
    # exit()
    print("=")

    # state_dict = torch.load("/work/yuxiang1234/backup/slue-qa-ckpt/QAReader-epoch=33-val_ff1=0.24.ckpt")['state_dict']
    # for n,p in state_dict.items():
    #     For encoder
    #     n = n.replace("reader.reader", "longformer")
    #     For qa
    #     n = n.replace("reader.QAhead", "qa_outputs")

    #     if n not in model.state_dict():
    #         print(n)
    #         print(f'{n} ----- parameter name not match!')
    #         continue
    #     if isinstance(p, nn.Parameter):
    #         print(f"load {n}")
    #         p = p.data
    #     model.state_dict()[n].copy_(p)
    # exit()
    print('[INFO]    loading data')
    
    train_dataset = SQADataset(data_dir, mode='train')
    dev_dataset = SQADataset(data_dir, mode='validation')

    print('[INFO]    loading done')

    train_loader = DataLoader(train_dataset, batch_size=training_args.per_gpu_train_batch_size, shuffle=True, pin_memory=True, collate_fn = collate_batch)
    dev_loader = DataLoader(dev_dataset, batch_size=training_args.per_gpu_eval_batch_size, shuffle=False,pin_memory=True, collate_fn = collate_batch)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collate_batch,
        # compute_metrics = compute_metrics
    )
    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))
    
        results.update(eval_output)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baseline.yaml', type=str)
    parser.add_argument('--exp_name', default='test', type=str)
    
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    print('[INFO]    Using config {}'.format(args.config))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_dir = config['data']['data_dir']
    main()
