# -*- coding: utf-8 -*-
# +
import re
from builtins import str as unicode
import heapq
import ast
import torch
from pandas import Interval
def text_preprocess(text):
    text = unicode(text)
    
    text = normalize_numbers(text)
    
    text = text.lower()
    text = text.replace("i.e.", "that is")
    text = text.replace("e.g.", "for example")
    text = text.replace(r"\%", "percent")
    text = re.sub("-", " ", text)
    text = re.sub("[^ a-z]", "", text)

    return text

# from g2p-en/expand.py
import inflect
import re

_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')


def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'    # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    else:
        return _inflect.number_to_words(num, andword='')


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


'''
metric calculation
'''
def compare(pred_start, pred_end, gold_start, gold_end):
    if pred_start >= pred_end: 
        overlap_start = 0
        overlap_end = 0
        Max = 0
        Min = 0
        no_overlap = True
    elif pred_end <= gold_start or pred_start >= gold_end:
        overlap_start = 0
        overlap_end = 0
        Max = 0
        Min = 0
        no_overlap = True
    elif gold_end == gold_start: 
        overlap_start = 0
        overlap_end = 0
        Max = 0
        Min = 0
        no_overlap = True
    else:
        no_overlap = False
        if pred_start <= gold_start:
            Min = pred_start
            overlap_start = gold_start
        else: 
            Min = gold_start
            overlap_start = pred_start

        if pred_end <= gold_end:
            Max = gold_end
            overlap_end = pred_end
        else: 
            Max = pred_end
            overlap_end = gold_end
        
    return overlap_start, overlap_end, Min, Max, no_overlap

def Frame_F1_score(pred_start, pred_end, gold_start, gold_end):
    overlap_start, overlap_end, Min, Max, no_overlap = compare(pred_start, pred_end, gold_start, gold_end)
    if no_overlap: 
        if pred_start == gold_start and pred_end == gold_end:
            F1 = 1
        else: 
            F1 = 0
    else: 
        Precision = (overlap_end - overlap_start) / (pred_end - pred_start)
        Recall = (overlap_end - overlap_start) / (gold_end - gold_start)
        F1 = 2 * Precision * Recall / (Precision + Recall)
    return F1

#pred_starts is the ground true
def Frame_F1_scores(pred_starts, pred_ends, gold_starts, gold_ends,labels):
    F1s = dict()
    total_diff = 0
    
    # print(labels)
    for label in labels:
        F1s[label] = []

    for GroundTruth_start, GroundTruth_end, gold_start, gold_end,label in zip(ast.literal_eval(pred_starts), ast.literal_eval(pred_ends), gold_starts, gold_ends,labels):
        #ignore the model correct answer negative example
        
        if(GroundTruth_start[0] == 0 and GroundTruth_end[0] == 0 and gold_start[0] == 0 and gold_end[0] == 0 ):
            continue

        temp_F1 = []
        min_len = min( len(GroundTruth_start),len(gold_start) )
        max_len = max( len(GroundTruth_start),len(gold_start) )
        F1_score = []

        #give all the combination between pred and gold pair
        for start_sec,end_sec in zip(gold_start, gold_end):
            for GroundTruth_start_sec,GroundTruth_end_sec in zip(GroundTruth_start,GroundTruth_end):
                F1 = Frame_F1_score(GroundTruth_start_sec,GroundTruth_end_sec,start_sec,end_sec)

                heapq.heappush( temp_F1,(-F1,start_sec,end_sec,GroundTruth_start_sec,GroundTruth_end_sec) )
        
        #select the overlap number between pred and gold
        exsit = set()
        count = 0
        while count != min_len:
            element = heapq.heappop(temp_F1)

            if( element[1:3] not in exsit):
                F1_score.append( -element[0] )
                exsit.add(element[1:3])
                count += 1

        #F1 score is 0 if there are excessive pair in pred or gold

        diff =  max_len - min_len
        total_diff += diff
        for i in range(diff):
            F1_score.append(0)
        
        F1s[label] += F1_score
    
    return F1s,total_diff

def aggregate_dev_result(metric):
    aggregate_result = []
    for i in range(len(metric)):
        aggregate_result.append(metric[i])
    return sum(aggregate_result) / len(aggregate_result)

def calc_overlap(pred_starts, pred_ends, gold_starts, gold_ends):
    x = [pred_starts, pred_ends]
    y = [gold_starts, gold_ends]
    if x[1] <= y[0] or x[0] >= y[1]:
        return 0.0, 0.0
    minest, maxest = min(x[0], y[0]), max(x[1], y[1])
    left, right = max(x[0], y[0]), min(x[1], y[1])
    try:
        aos = (right - left) / (maxest - minest)
        precision = (right - left) / (x[1] - x[0])
        recall = (right - left) / (y[1] - y[0])
        f1 = float((2 * precision * recall) / (precision + recall))
    except:
        print(right, left, maxest, minest)
    return f1, aos

#Carefully check your context_end. You may want different context_end.
def _get_best_indexes(probs, context_offset,context_end,k):
    # do not need to consider indexes for question and determine the end of consideration.
    probs = probs[context_offset:context_offset + context_end]

    #top-k method
    if(k < len(probs)):
        top_values, top_indices = torch.topk(probs, k)
    else:
        top_values, top_indices = torch.topk(probs, len(probs))

    best_indexes = top_indices + context_offset

    return best_indexes

def post_process_prediction(start_prob, end_prob,context_offset,context_len,negative_start,negative_end,threshold,max_answer_length,weight = 0.6):
    
    start_prob = start_prob.squeeze()
    end_prob = end_prob.squeeze()
    
    #Carefully check your context_end
    start_indexes = _get_best_indexes(start_prob,context_offset,context_len,10)
    end_indexes = _get_best_indexes(end_prob,context_offset,context_len,10)

    final_start_indexes = []
    final_end_indexes = []

    negative_score = start_prob[negative_start] + end_prob[negative_end]

    prelim_predictions = []

    for start_index in start_indexes:
        for end_index in end_indexes:
            
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue

            predict = {
                        'start_prob': start_prob[start_index],
                        'end_prob': end_prob[end_index],
                        'start_idx': start_index.item(), 
                        'end_idx': end_index.item(),
                        'score': start_prob[start_index] + end_prob[end_index]
                      }

            prelim_predictions.append(predict)

    prelim_predictions = sorted(prelim_predictions, 
                                key=lambda x: x['score'],
                                reverse=True)
    
    for candidate in prelim_predictions:
        if(candidate['score'] >= threshold and candidate['score'] >= negative_score):
            final_start_indexes.append(candidate['start_idx'])
            final_end_indexes.append(candidate['end_idx'])

    if(len(final_start_indexes) == 0):
        final_start_indexes.append(negative_start)
        final_end_indexes.append(negative_end)
    
    output_start = []
    output_end = []

    for start,end in zip(final_start_indexes,final_end_indexes):
        candidate_pair = Interval(start,end,closed='both')

        overlapping = False

        for start_,end_ in zip(output_start,output_end):
            decide_pair = Interval(start_,end_,closed='both')

            if decide_pair.overlaps(candidate_pair):
                overlapping = True
                break
        
        if overlapping == False:
            output_start.append(start)
            output_end.append(end)

    return output_start,output_end

def process_overlapping(start_probs,end_probs,starts,ends,context_begins,weight = 0.6):
    total = []
    i = 0

    #gather model output
    for start,end in zip(starts,ends):
        for start_index,end_index in zip(start,end):
            if start_index == 0 and end_index == 0:
                break
            else:
                score = start_probs[i][start_index] + end_probs[i][end_index]
                total.append((start_index - context_begins[i],end_index - context_begins[i], i,score))

        i += 1

    total.sort(key = lambda x:x[3])

    outputs = []

    # filter
    for answer in total:
        candidate_pair = Interval(answer[0].item(),answer[1].item(),closed='both')

        overlap = False
        for output in outputs:
            output_pair = Interval(output[0].item(),output[1].item(),closed='both')

            if output_pair.overlaps(candidate_pair):
                overlap = True
                break
        
        if overlap == False:
            outputs.append(answer)
    
    #output
    final_starts = [[] for _ in range(7)]
    final_ends = [[] for _ in range(7)]

    for output in outputs:
        label_index = output[2]

        start_ = output[0] + context_begins[label_index]
        end_ = output[1] + context_begins[label_index]

        final_starts[label_index].append(start_.item())
        final_ends[label_index].append(end_.item())
    
    for final_start,final_end in zip(final_starts,final_ends):
        if not final_start:
            final_start.append(0)
            final_end.append(0)
    
    return final_starts, final_ends

def find_overlapframe(pred_starts, pred_ends, gold_starts, gold_ends,labels,overlap_frame,predict_frame,groundtruth_frame):

    for pred_start, pred_end, gold_start, gold_end,label in zip(ast.literal_eval(pred_starts), ast.literal_eval(pred_ends), gold_starts, gold_ends,labels):
        
        candidate_overlap = []
        for start_sec,end_sec in zip(gold_start, gold_end):
            for GroundTruth_start_sec,GroundTruth_end_sec in zip(pred_start,pred_end):
                
                overlap_start, overlap_end, Min, Max, no_overlap = compare(GroundTruth_start_sec,GroundTruth_end_sec,start_sec,end_sec)
                heapq.heappush( candidate_overlap,(-(overlap_end - overlap_start),start_sec,end_sec))
        
        count = 0
        exist = set()
        final_overlap = []

        while count != len(gold_start):
            element = heapq.heappop(candidate_overlap)

            if element[1:3] not in exist:
                exist.add(element[1:3])
                count += 1
                final_overlap.append(-element[0])
        
        overlap_frame[label] += final_overlap

        for start_sec,end_sec in zip(gold_start, gold_end):
            predict_frame[label].append(end_sec - start_sec)
        
        for GroundTruth_start_sec,GroundTruth_end_sec in zip(pred_start,pred_end):
            groundtruth_frame[label].append(GroundTruth_end_sec - GroundTruth_start_sec)
    
    return

def calculate_FF1(overlap_frame, predict_frame, groundtruth_frame, Combined_label):
    
    Precision = []
    Recall = []
    FF1 = []

    total_overlapframe = 0
    total_predict_frame = 0
    total_groundtruth_frame = 0

    for idx,label in enumerate(Combined_label):
        sum_overlap_frame = sum(overlap_frame[label])
        sum_predict_frame = sum(predict_frame[label])
        sum_groundtruth_frame = sum(groundtruth_frame[label])

        total_overlapframe += sum_overlap_frame
        total_predict_frame += sum_predict_frame
        total_groundtruth_frame += sum_groundtruth_frame

        Precision.append(sum_overlap_frame /  sum_groundtruth_frame)
        Recall.append(sum_overlap_frame /  sum_predict_frame)
        FF1.append(2 * Precision[idx] * Recall[idx] / (Precision[idx] + Recall[idx]))
    
    TotalPrecision = total_overlapframe / total_groundtruth_frame
    TotalRecall = total_overlapframe / total_predict_frame
    TotalFF1 = 2 * TotalPrecision * TotalRecall / (TotalPrecision + TotalRecall)

    return TotalPrecision, TotalRecall, TotalFF1, Precision, Recall, FF1
    
    





    

    
