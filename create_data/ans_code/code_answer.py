# +
import pandas as pd
import numpy as np
import os
import math
from tqdm import tqdm
from utils import find_code_answer

def main():
    mode = 'fine-tune'
    file_dir = '/work/f776655321/DUAL-textless-NER/code-data/code/' + mode +'/'
    df = pd.read_csv('/work/f776655321/DUAL-textless-NER/code-data/' + mode +'_ans.csv')
    start = df['start'].values
    end = df['end'].values
    passage = df['context_id'].values
    labels = df['label'].values

    code_start = []
    code_end = []

    new_context_ids = []
    new_labels = []
    
    for start_sec, end_sec, context_id,label in tqdm(zip(start, end, passage,labels)):
        
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


    NewDf.to_csv('/work/f776655321/DUAL-textless-NER/code-data/' + mode + '_code_ans.csv',index=False)

if __name__ == "__main__":
    main()