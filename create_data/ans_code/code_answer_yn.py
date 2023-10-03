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
        # TODO
        if mode == 'train':
            df = pd.read_csv('code-data/' + mode + f'_ans_negative_yn.csv')
        else: 
            df = pd.read_csv('code-data/' + mode + f'_ans_yn.csv')

        start = df['start'].values
        end = df['end'].values
        passage = df['context_id'].values
        labels = df['label'].values
        yes_or_no = df["label_text"].values

        code_start = []
        code_end = []
        context_len = []
        new_context_ids = []
        new_labels = []
        answers = []
        label_cnt = defaultdict(int)
        data_by_label = defaultdict(list)

        action_list = ["question_check", "question_repeat", "question_general", "answer_agree", "answer_dis", "answer_general", "apology", "thanks", "acknowledge", "statement_open", "statement_close", "statement_problem", "statement_instruct", "statement_general", "backchannel", "disfluency", "self", "other"]
        action_code = {}
        action_cnt = {}
        cumulative_cnt = []
        selection_list = ["yes", "no"]
        root_dir = "code-data/question-code-DAC-yn"
        for idx, answer in enumerate(selection_list):
            code = np.loadtxt(os.path.join(root_dir, answer + '.code'))
            cnt = len(code)
            action_code[answer] = code 
            action_cnt[answer] = len(code)
            if idx == 0:
                cumulative_cnt.append(cnt)
            else: 
                cumulative_cnt.append(cnt + cumulative_cnt[idx - 1])
        
        for start_sec, end_sec, context_id,label, yn in tqdm(zip(start, end, passage,labels, yes_or_no)):
            context_code = np.loadtxt(os.path.join(file_dir, context_id + '.code'))
            if context_code.shape == ():
                # print(code)
                context_code = np.expand_dims(context_code, axis=-1)
            
            if start_sec == -1 and end_sec == -1:
                code_start.append(len(context_code) + cumulative_cnt[0])
                code_end.append(len(context_code) + cumulative_cnt[1] - 1)
                answers.append(yn)      
                new_context_ids.append(context_id)
                new_labels.append(label)  
            else:
                code_start.append(len(context_code))
                code_end.append(len(context_code) + cumulative_cnt[0] - 1)
                answers.append(yn)      
                new_context_ids.append(context_id)
                new_labels.append(label)                  
            


        NewDf = pd.DataFrame()

        NewDf['context_id'] = new_context_ids
        NewDf['label'] = new_labels
        NewDf['code_start'] = code_start
        NewDf['code_end'] = code_end
        NewDf['answer'] = answers
        # TODO
        if mode == "train":
            NewDf.to_csv('code-data/' + mode + f'_code_ans_negative_yn.csv',index=False)
        else: 
            NewDf.to_csv('code-data/' + mode + f'_code_ans_yn.csv',index=False)


if __name__ == "__main__":
    main()