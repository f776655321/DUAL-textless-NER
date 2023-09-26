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
        ds = load_from_disk(f"slue-sa/{mode}.hf")

        #for output csv
        ids = []
        actions = []
        start_seconds = []
        end_seconds = []
        action_text = []
        output_dir = 'code-data-sa/'
        
        K = 2
        if mode == 'train':
            # output_file = mode + f'_ans_sampling_positive_{K}.csv'
            output_file = mode + f'_ans_balance.csv'
            # output_file = mode + f'_ans_sampling_negative_{K}.csv'
            # output_file = mode + f'_ans_sampling_negative_slue_{K}.csv'
        else: 
            output_file = mode + "_ans.csv"
            # output_file = mode + "_ans_slue.csv"

        #for New label
        action_list = ["positive", "neutral", "negative"]
        cumulative_time = []
        root_path = "code-data-sa/question-prompts-SA"
        # root_path = "/work/yuxiang1234/DUAL-textless-DAC-2/code-data/question-prompts-dac-slue"
        for idx, action in enumerate(action_list):
            wav = MP3(os.path.join(root_path, action + ".mp3")).info
            if idx == 0:
                cumulative_time.append(wav.length)
            else: 
                cumulative_time.append(wav.length + cumulative_time[idx - 1])

        # measure the percentage of positive_example
        for line in tqdm(ds):
            label = line["sentiment"].lower()
            # for label in labels:
            positive_cnt[label] += 1

        positive_ratio = {key: value / len(ds)  for key, value in positive_cnt.items()}
        
        print(positive_ratio)


        data_by_sentiment = defaultdict(list)
        for line in tqdm(ds):
            title = line["id"]
            utt_index = line["speaker_id"]
            text = line["normalized_text"]            
            audio = line["audio"]["array"]
            start_second = line["start_second"]
            end_second = line["end_second"]

            label = line["sentiment"].lower()
            id = f"{title}_{utt_index}"
            
            for i, action in enumerate(action_list):
                
                # ner_tags is none
                if action  == label:
                    ids.append(id)
                    action_text.append(action)
                    actions.append(action)
                    start_seconds.append(0)
                    end_seconds.append(0)
                    data_by_sentiment[action].append(id)
        max_num = 0
        max_action = None
        for action in action_list:
            if len(data_by_sentiment[action]) > max_num:
                max_num = len(data_by_sentiment[action])
                max_action = action

        for action in action_list:
            if max_action == action:
                continue            
            samples = random.choices(data_by_sentiment[action], k= max_num - len(data_by_sentiment[action]))
            print(samples)
            for s in samples:
                ids.append(s)
                action_text.append(action)
                actions.append(action)
                start_seconds.append(0)
                end_seconds.append(0)           


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