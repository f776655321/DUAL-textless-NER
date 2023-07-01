import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm

def add_result(result,label2index,label,code):
    if(code != -1):
        result[label2index[label]].append(code)
    else:
        result[label2index[label]].append(0)

# merge the answer of the same context_id
def merge_answer():
    data_dir = '../../code-data'
    mode = 'fine-tune'
    input_file = mode + '_code_ans.csv'

    df = pd.read_csv(os.path.join(data_dir,input_file))

    context_ids = df['context_id'].values
    code_starts = df['code_start'].values
    code_ends = df['code_end'].values
    labels = df['label'].values

    Combined_label = ['PLACE','QUANT','ORG','WHEN','NORP','PERSON','LAW']
    label2index = {'PLACE' : 0,'QUANT' : 1,'ORG': 2,'WHEN': 3,'NORP': 4,'PERSON': 5,'LAW': 6}

    final_starts = []
    final_ends = []
    final_context_id = []

    prev = None

    for context_id,label,code_start,code_end in tqdm(zip(context_ids,labels,code_starts,code_ends)):

        if(prev == None):
            final_context_id.append(context_id)
            start_result = [[] for _ in Combined_label]
            end_result = [[] for _ in Combined_label]
            
            add_result(start_result,label2index,label,code_start)
            add_result(end_result,label2index,label,code_end)

            prev = context_id
        elif(prev != context_id):
            final_context_id.append(context_id)

            for start,end in zip(start_result,end_result):
                if(len(start) == 0 and len(end) == 0):
                    start.append(0)
                    end.append(0)

            final_starts.append(start_result)
            final_ends.append(end_result)       

            start_result = [[] for _ in Combined_label]
            end_result = [[] for _ in Combined_label]
            
            add_result(start_result,label2index,label,code_start)
            add_result(end_result,label2index,label,code_end)

            prev = context_id
        else:
    
            add_result(start_result,label2index,label,code_start)
            add_result(end_result,label2index,label,code_end)

    for start,end in zip(start_result,end_result):
        if(len(start) == 0 and len(end) == 0):
            start.append(0)
            end.append(0)
    final_starts.append(start_result)
    final_ends.append(end_result)

    return final_context_id,final_starts,final_ends

def idx2frame(start_frame_idx, end_frame_idx,context_cnt):
    context_cnt = torch.tensor(context_cnt.squeeze())

    start_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:start_frame_idx].size()), context_cnt[:start_frame_idx])
    
    end_frame_idx = torch.repeat_interleave(torch.ones(context_cnt[:end_frame_idx].size()), context_cnt[:end_frame_idx])
    
    start_idx, end_idx = torch.sum(start_frame_idx), torch.sum(end_frame_idx)
    return start_idx.item(),end_idx.item()


def main():
    context_ids,starts,ends = merge_answer()
    mode = 'fine-tune'
    data_dir = '../../code-data'
    code_passage_dir = os.path.join(data_dir, 'code',mode)

    for context_id,start_array,end_array in tqdm(zip(context_ids,starts,ends)):
        context_cnt = np.loadtxt(os.path.join(code_passage_dir, context_id+'.cnt')).astype(int)
        for start,end in zip(start_array,end_array):
            if(start[0] == 0 and end[0] == 0):
                pass
            else:
                for i in range(len(start)):
                    start[i],end[i]= idx2frame(start[i],end[i],context_cnt)
    NewDf = pd.DataFrame()

    NewDf['context_id'] = context_ids
    NewDf['start'] = starts
    NewDf['end'] = ends

    NewDf.to_csv(os.path.join(data_dir,mode +'_frame_inference.csv'), index=False)

if __name__ == '__main__':
    main()

