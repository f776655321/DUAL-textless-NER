import pandas as pd
import os
import json

import torch
import torchaudio

import random
from tqdm import tqdm
import ast

from .utils import force_align,get_ans_time,random_choice_without_repeat


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

    data_dir = '/work/f776655321/DUAL-textless-NER'

    with open(os.path.join(data_dir, mode+'-hash2context.json')) as f:
        h2q = json.load(f)

    mode = 'fine-tune'

    audio_dir = '/work/f776655321/DUAL-textless-NER/slue-voxpopuli/' + mode + '/'

    #import slue data
    sluedf = pd.read_csv('/work/f776655321/DUAL-textless-NER/slue-voxpopuli/slue-voxpopuli_' + mode +'.tsv',sep='\t')

    normalize_text = sluedf['normalized_text'].values
    normalized_ner = sluedf['normalized_ner'].values

    old_context_id = sluedf['id'].values

    new_context_id = sluedf['id'].apply(lambda x: h2q[x]).values


    #for output csv
    ans_context_id = []
    label = []
    start = []
    end = []
    label_text = []
    output_dir = '../../code-data/'
    output_file = mode + '_ans.csv'

    #WAV2VEC2 need words to be separated by '|'
    length = len(normalize_text)
    for i in range(length):
        normalize_text[i] =  normalize_text[i].replace(' ', '|')

    #for New label
    NER_Label_dict = {'GPE':'PLACE','LOC':'PLACE','CARDINAL': 'QUANT','ORDINAL':'QUANT','MONEY':'QUANT','PERCENT':'QUANT','QUANTITY':'QUANT','ORG':'ORG','DATE':'WHEN','TIME':'WHEN','NORP':'NORP','PERSON':'PERSON','LAW':'LAW'}

    Combined_label = ['PLACE','QUANT','ORG','WHEN','NORP','PERSON','LAW']

    #used to check if force alignment fail
    not_alignable = []

    for i , ner_tags in enumerate(tqdm(normalized_ner)):
        
        # ner_tags is none
        if(isinstance(ner_tags, float) == True):
            tag = random.choice(Combined_label)
            label_text.append("none")
            ans_context_id.append(new_context_id[i])
            label.append(tag)
            start.append(0)
            end.append(0)
        else:
        
            SPEECH_FILE = old_context_id[i]

            new_normalize_text = normalize_text[i].replace(';','').replace('.','').replace('?','').replace('!','')

            word_segments,waveform_size,trellis_size = force_align(model,audio_dir,SPEECH_FILE,dictionary,new_normalize_text)

            ner_tags = ast.literal_eval(ner_tags)

            find_time= set()
            
            n = len(ner_tags)

            exist_label = set()

            for ner_tag in ner_tags:

                if(ner_tag[0] not in NER_Label_dict):
                    n -= 1
                else:

                    ans_context_id.append(new_context_id[i])

                    label.append(NER_Label_dict[ner_tag[0]])

                    exist_label.add(NER_Label_dict[ner_tag[0]])
                    
                    while(ner_tag[1] != 0 and normalize_text[i][ner_tag[1] - 1] != '|'):
                        ner_tag[1] -= 1

                    target = normalize_text[i][ner_tag[1]:ner_tag[1] + ner_tag[2]]

                    target = target.strip('|')

                    change_target = target.replace('|',' ')

                    label_text.append(change_target)

                    target = target.split('|')

                    start_time,end_time = get_ans_time(target, word_segments,waveform_size,trellis_size,find_time,sample_rate)

                    start.append(round(start_time,3))

                    end.append(round(end_time,3))

                    if start_time == 0 and end_time == 0:
                        not_alignable.append(change_target)

            notexist_label = [tag for tag in Combined_label if tag not in exist_label]

            n = 0 if mode == "dev" else int(n/2)
            tags = random_choice_without_repeat(notexist_label,n)

            for tag in tags:
                ans_context_id.append(new_context_id[i])
                label_text.append("none")
                label.append(tag)
                start.append(0)
                end.append(0)

    df = pd.DataFrame()
    df['context_id'] = ans_context_id
    df['label'] = label
    df['label_text'] = label_text
    df['start'] = start
    df['end'] = end

    df.to_csv(os.path.join(output_dir,output_file), index=False)

    print(not_alignable)

if __name__ == "__main__":
    main()