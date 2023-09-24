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

    #############wav2vec2 model############################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.random.manual_seed(0)

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
    model = bundle.get_model().to(device)

    sample_rate = bundle.sample_rate

    labels = bundle.get_labels()

    dictionary = {c.lower(): i for i, c in enumerate(labels)}

    ######################################################


    random.seed(0)
    for mode in ["train", "validation", "test"]:
        positive_cnt = defaultdict(int)
        negative_cnt = defaultdict(int)
        ds = load_from_disk(f"slue-dac/{mode}.hf")

        #for output csv
        ids = []
        actions = []
        start_seconds = []
        end_seconds = []
        action_text = []
        output_dir = 'code-data/'
        
        K = 2
        if mode == 'train':
            # output_file = mode + f'_ans_sampling_positive_{K}.csv'
            output_file = mode + f'_ans.csv'
            # output_file = mode + f'_ans_sampling_negative_{K}.csv'
            # output_file = mode + f'_ans_sampling_negative_slue_{K}.csv'
        else: 
            output_file = mode + "_ans.csv"
            # output_file = mode + "_ans_slue.csv"

        #for New label
        action_list = ["question_check", "question_repeat", "question_general", "answer_agree", "answer_dis", "answer_general", "apology", "thanks", "acknowledge", "statement_open", "statement_close", "statement_problem", "statement_instruct", "statement_general", "backchannel", "disfluency", "self", "other"]
        cumulative_time = []
        root_path = "code-data/question-prompts-DAC"
        # root_path = "/work/yuxiang1234/DUAL-textless-DAC-2/code-data/question-prompts-dac-slue"
        for idx, action in enumerate(action_list):
            wav = MP3(os.path.join(root_path, action + ".mp3")).info
            if idx == 0:
                cumulative_time.append(wav.length)
            else: 
                cumulative_time.append(wav.length + cumulative_time[idx - 1])

        double_action = ["answer_agree", "question_repeat"]
        K_action = ["backchannel", 'statement_instruct', "self", "apology"]
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
                    action_text.append(text)
                    actions.append(action)
                    start_seconds.append(0)
                    end_seconds.append(float(duration) / 1000.0 + cumulative_time[action_list.index(action)])
                    # end_second.append(float(duration) / 1000.0)
                    # if mode == "train" and action_text in double_action:
                    #     for i in range(3):
                    #         ids.append(id)
                    #         action_text.append(text)
                    #         actions.append(action)
                    #         start_seconds.append(0)
                    #         end_seconds.append(float(duration) / 1000.0 + cumulative_time[action_list.index(action)])
                    # if mode == "train" and action_text in K_action:
                    #     for i in range(10):
                    #         ids.append(id)
                    #         action_text.append(text)
                    #         actions.append(action)
                    #         start_seconds.append(0)
                    #         end_seconds.append(float(duration) / 1000.0 + cumulative_time[action_list.index(action)])                    
                # else: 
                #     if mode == 'train':
                #         select = random.random()
                #         # if select < positive_ratio[action] * K:
                #         if True:
                #             ids.append(id)
                #             action_text.append(text)
                #             actions.append(action)
                #             start_seconds.append(0)
                #             end_seconds.append(0)
                #             negative_cnt[action] += 1


        negative_ratio = {key: value / len(ds)  for key, value in negative_cnt.items()}

        for key in negative_ratio:
            print(f"pos: {positive_ratio[key]}, neg: {negative_ratio[key]}")

        df = pd.DataFrame()
        df['context_id'] = ids
        df['label'] = actions
        df['label_text'] = action_text
        df['start'] = start_seconds
        df['end'] = end_seconds

        df.to_csv(os.path.join(output_dir,output_file), index=False)


if __name__ == "__main__":
    main()