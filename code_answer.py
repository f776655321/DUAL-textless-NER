# +
import pandas as pd
import numpy as np
import os
import math
from tqdm import tqdm

def find_code_answer(start_sec, end_sec, context_id,context_code,context_cnt):
    # one frame equals  to 20ms
    start_ind = start_sec / 0.02
    end_ind = end_sec / 0.02
    context_cnt_cum = np.cumsum(context_cnt)
    new_start_ind, new_end_ind = None, None
    # print(start_ind, end_ind)
    prev = 0
    for idx, cum_idx in enumerate(context_cnt_cum): 
        
        if cum_idx >= start_ind and new_start_ind is None:
            if abs(start_ind - prev) <= abs(cum_idx - start_ind):
                new_start_ind = idx - 1
                if(new_start_ind < 0 and (start_ind != 0 or end_ind != 0)):
                    new_start_ind = 0
            else:
                new_start_ind = idx
        if cum_idx >= end_ind and new_end_ind is None:
            if abs(end_ind - prev) <= abs(cum_idx - end_ind):
                new_end_ind = idx - 1
                if(new_end_ind < 0 and (start_ind != 0 or end_ind != 0)):
                    new_end_ind = 0
            else:
                new_end_ind = idx
        prev = cum_idx
    if new_start_ind == None: 
        new_start_ind = idx
    if new_end_ind == None: 
        new_end_ind = idx
    
    return new_start_ind, new_end_ind



mode = 'fine-tune'
file_dir = '/work/f776655321/DUAL-textless-NER/code-data/SpecAugment-code-km512/' + mode +'/'
df = pd.read_csv('/work/f776655321/DUAL-textless-NER/code-data/' + mode +'_ans_with_n_and_r.csv')
start = df['start'].values
end = df['end'].values
passage = df['context_id'].values
labels = df['label'].values

code_start = []
code_end = []

new_context_ids = []
new_labels = []
for start_sec, end_sec, context_id,label in tqdm(zip(start, end, passage,labels)):
#     context_code = np.loadtxt(os.path.join(file_dir, context_id+'.code'))
#     context_cnt = np.loadtxt(os.path.join(file_dir, context_id+'.cnt'))
    context_code = np.loadtxt(os.path.join(file_dir, context_id + '.code'))
    context_cnt = np.loadtxt(os.path.join(file_dir, context_id + '.cnt'))

    new_context_ids.append(context_id)
    new_labels.append(label)

    new_start_ind,new_end_ind =  find_code_answer(start_sec, end_sec, context_id,context_code,context_cnt)
    
    code_start.append(new_start_ind)
    code_end.append(new_end_ind)
    
    #again for augment code
    context_id = context_id + "_augment"

    context_code = np.loadtxt(os.path.join(file_dir, context_id + '.code'))
    context_cnt = np.loadtxt(os.path.join(file_dir, context_id + '.cnt'))

    new_context_ids.append(context_id)
    new_labels.append(label)

    new_start_ind,new_end_ind =  find_code_answer(start_sec, end_sec, context_id,context_code,context_cnt)
    
    code_start.append(new_start_ind)
    code_end.append(new_end_ind)

NewDf = pd.DataFrame()

NewDf['context_id'] = new_context_ids
NewDf['label'] = new_labels
NewDf['code_start'] = code_start
NewDf['code_end'] = code_end


NewDf.to_csv('/work/f776655321/DUAL-textless-NER/code-data/' + mode + '_code_aug_km512_ans.csv',index=False)