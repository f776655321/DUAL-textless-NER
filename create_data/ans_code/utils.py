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