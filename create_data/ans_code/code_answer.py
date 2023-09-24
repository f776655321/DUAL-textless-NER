# +
import pandas as pd
import numpy as np
import os
import math
import random
from tqdm import tqdm
from collections import defaultdict
from utils import find_code_answer

def main():
    K = 2
    random.seed(0)
    for mode in ["train", "validation", "test"]:
        file_dir = 'code-data/code/' + mode + '/'
        # file_dir = 'code-data/code-slue/' + mode + '/'
        # TODO
        if mode == 'train':
            # df = pd.read_csv('code-data/' + mode + f'_ans_sampling_positive_{K}.csv')
            df = pd.read_csv('code-data/' + mode + f'_ans.csv')
            # df = pd.read_csv('code-data/' + mode + f'_ans_sampling_negative_{K}.csv')
            # df = pd.read_csv('code-data/' + mode + f'_ans_sampling_negative_slue_{K}.csv')
        else: 
            df = pd.read_csv('code-data/' + mode + f'_ans.csv')
            # df = pd.read_csv('/work/yuxiang1234/DUAL-textless-DAC-2/code-data/' + mode + f'_ans_slue.csv')

        start = df['start'].values
        end = df['end'].values
        passage = df['context_id'].values
        labels = df['label'].values

        code_start = []
        code_end = []
        context_len = []
        new_context_ids = []
        new_labels = []
        label_cnt = defaultdict(int)
        data_by_label = defaultdict(list)

        action_list = ["question_check", "question_repeat", "question_general", "answer_agree", "answer_dis", "answer_general", "apology", "thanks", "acknowledge", "statement_open", "statement_close", "statement_problem", "statement_instruct", "statement_general", "backchannel", "disfluency", "self", "other"]
        action_code = {}
        action_cnt = {}
        cumulative_cnt = []
        # root_dir = "code-data/question-code-dac-slue"
        root_dir = "code-data/question-code-DAC"
        for idx, action in enumerate(action_list):
            code = np.loadtxt(os.path.join(root_dir, action + '.code'))
            cnt = len(code)
            action_code[action] = code 
            action_cnt[action] = len(code)
            if idx == 0:
                cumulative_cnt.append(cnt)
            else: 
                cumulative_cnt.append(cnt + cumulative_cnt[idx - 1])
        
        for start_sec, end_sec, context_id,label in tqdm(zip(start, end, passage,labels)):
            context_code = np.loadtxt(os.path.join(file_dir, context_id + '.code'))
            if context_code.shape == ():
                # print(code)
                context_code = np.expand_dims(context_code, axis=-1)
            
            if start_sec == 0 and end_sec == 0:
                code_start.append(0)
                code_end.append(0)
                context_len.append(len(context_code))      
                new_context_ids.append(context_id)
                new_labels.append(label)  
                continue 
            
            new_context_ids.append(context_id)
            new_labels.append(label)
            # TODO
            # new_start_ind, new_end_ind =  find_code_answer(start_sec, end_sec, context_id,context_code,context_cnt)
            # start position start from 0
            new_start_ind = int(len(context_code) + cumulative_cnt[action_list.index(label)] - action_cnt[label] + 1) - 1
            new_end_ind = int(len(context_code) + cumulative_cnt[action_list.index(label)] ) - 1

            code_start.append(new_start_ind)
            code_end.append(new_end_ind)
            label_cnt[label] += 1
            context_len.append(len(context_code))
            data_by_label[label].append((context_id, label, new_start_ind, new_end_ind, len(context_code)))

        # for key in label_cnt.keys():
        #     while label_cnt[key] < 5000:
        #         idx = random.randint(0, len(data_by_label[key]) - 1)
        #         print(idx)
        #         repeat_data = data_by_label[key][idx]
        #         new_context_ids.append(repeat_data[0])
        #         new_labels.append(repeat_data[1])
        #         code_start.append(repeat_data[2])
        #         code_end.append(repeat_data[3])
        #         context_len.append(repeat_data[4])
        #         label_cnt[key] += 1

        NewDf = pd.DataFrame()

        NewDf['context_id'] = new_context_ids
        NewDf['label'] = new_labels
        NewDf['code_start'] = code_start
        NewDf['code_end'] = code_end
        NewDf['context_len'] = context_len
        # TODO
        if mode == "train":
            # NewDf.to_csv('code-data/' + mode + f'_code_ans_sampling_positive_{K}.csv',index=False)
            # NewDf.to_csv('code-data/' + mode + f'_code_ans_sampling_positive_{K}.csv',index=False)
            NewDf.to_csv('code-data/' + mode + f'_code_ans.csv',index=False)
        else: 
            # NewDf.to_csv('code-data/' + mode + f'_code_ans_slue.csv',index=False)
            NewDf.to_csv('code-data/' + mode + f'_code_ans.csv',index=False)


if __name__ == "__main__":
    main()