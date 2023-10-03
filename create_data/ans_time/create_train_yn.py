import pandas as pd
import os
import json
from collections import defaultdict

import torch
import torchaudio
import string
import mutagen

import random
from tqdm import tqdm
import ast
from datasets import load_from_disk
from mutagen.mp3 import MP3

from utils import force_align_from_data, get_ans_time, random_choice_without_repeat, remove_punctuation

def main():

    random.seed(0)
    for mode in ["train", "validation", "test"]:
        positive_cnt = defaultdict(int)
        negative_cnt = defaultdict(int)
        ds = load_from_disk(f"/work/yuxiang1234/backup/slue-dac/{mode}.hf")

        #for output csv
        ids = []
        answers = []
        start_seconds = []
        end_seconds = []
        answer_text = []
        output_dir = 'code-data/'
        
        K = 2
        if mode == 'train':
            output_file = mode + f'_ans_negative_yn.csv'
        else: 
            output_file = mode + "_ans_yn.csv"

        #for New label
        action_list = ["question_check", "question_repeat", "question_general", "answer_agree", "answer_dis", "answer_general", "apology", "thanks", "acknowledge", "statement_open", "statement_close", "statement_problem", "statement_instruct", "statement_general", "backchannel", "disfluency", "self", "other"]
        cumulative_time = []
        root_path = "code-data/question-prompts-DAC-yn"
        selection_list = ["yes", "no"]
        for idx, action in enumerate(selection_list):
            wav = MP3(os.path.join(root_path, action + ".mp3")).info
            if idx == 0:
                cumulative_time.append(wav.length)
            else: 
                cumulative_time.append(wav.length + cumulative_time[idx - 1])

        # measure the percentage of positive_example
        for line in tqdm(ds):
            labels = line["dialog_acts"]
            for label in labels:
                positive_cnt[label] += 1

        positive_ratio = {key: value / len(ds)  for key, value in positive_cnt.items()}
        
        print(positive_ratio)
        
        for line in tqdm(ds):
            title = line["issue_id"]
            utt_index = line["utt_index"]
            text = line["text"]            
            duration = line["duration_ms"]

            labels = line["dialog_acts"]
            id = f"{title}_{utt_index}"
            
            for i, action in enumerate(action_list):
                
                # ner_tags is none
                if action in labels:
                    ids.append(id)
                    answer_text.append("yes")
                    answers.append(action)
                    start_seconds.append(0)
                    end_seconds.append(0)
                else: 
                    if mode == 'train':
                        select = random.random()
                        if select < positive_ratio[action] * K:
                        # if True:
                            ids.append(id)
                            answer_text.append("no")
                            answers.append(action)
                            start_seconds.append(-1)
                            end_seconds.append(-1)
                            negative_cnt[action] += 1
                    elif mode == "validation": 
                        ids.append(id)
                        answer_text.append("no")
                        answers.append(action)
                        start_seconds.append(-1)
                        end_seconds.append(-1)
                        negative_cnt[action] += 1

        negative_ratio = {key: value / len(ds)  for key, value in negative_cnt.items()}

        for key in negative_ratio:
            print(f"pos: {positive_ratio[key]}, neg: {negative_ratio[key]}")

        df = pd.DataFrame()
        df['context_id'] = ids
        df['label'] = answers
        df['label_text'] = answer_text
        df['start'] = start_seconds
        df['end'] = end_seconds

        df.to_csv(os.path.join(output_dir,output_file), index=False)


if __name__ == "__main__":
    main()