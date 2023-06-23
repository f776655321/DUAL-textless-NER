# -*- coding: utf-8 -*-
# +
import re
from builtins import str as unicode
import heapq
import ast

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

    for pred_start, pred_end, gold_start, gold_end,label in zip(ast.literal_eval(pred_starts), ast.literal_eval(pred_ends), gold_starts, gold_ends,labels):
        #ignore the model correct answer negative example
        
        if(pred_start[0] == 0 and pred_end[0] == 0 and gold_start[0] == 0 and gold_end[0] == 0 ):
            continue

        temp_F1 = []
        min_len = min( len(pred_start),len(gold_start) )
        max_len = max( len(pred_start),len(gold_start) )
        F1_score = []
        #give all the combination between pred and gold pair
        for start_sec,end_sec in zip(gold_start, gold_end):
            for ground_start_sec,ground_end_sec in zip(pred_start,pred_end):
                F1 = Frame_F1_score(ground_start_sec,ground_end_sec,start_sec,end_sec)
                heapq.heappush( temp_F1,(-F1,start_sec,end_sec,ground_start_sec,ground_end_sec) )
        
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

def AOS_score(pred_start, pred_end, gold_start, gold_end):
    overlap_start, overlap_end, Min, Max, no_overlap = compare(pred_start, pred_end, gold_start, gold_end)

    if no_overlap:
        if(pred_start == gold_start and pred_end == gold_end):
            AOS = 1
        else:
            AOS = 0
    else: 
        AOS = (overlap_end - overlap_start) / (Max - Min)
    return AOS

def AOS_scores(pred_starts, pred_ends, gold_starts, gold_ends,labels):
    AOSs = dict()
    for label in labels:
        AOSs[label] = []
    for pred_start, pred_end, gold_start, gold_end,label in zip(ast.literal_eval(pred_starts), ast.literal_eval(pred_ends), gold_starts, gold_ends,labels):
        if(pred_start[0] == 0 and pred_end[0] == 0 and gold_start[0] == 0 and gold_end[0] == 0 ):
            continue
        temp_AOS = []
        min_len = min( len(pred_start),len(gold_start) )
        max_len = max( len(pred_start),len(gold_start) )
        AOSscore = []

        #give all the combination between pred and gold pair
        for start_sec,end_sec in zip(gold_start, gold_end):
            for ground_start_sec,ground_end_sec in zip(pred_start,pred_end):
                AOS = AOS_score(ground_start_sec,ground_end_sec,start_sec,end_sec)
                heapq.heappush( temp_AOS,(-AOS,start_sec,end_sec,ground_start_sec,ground_end_sec) )
        
        #select the overlap number between pred and gold
        exsit = set()
        count = 0
        while count != min_len:
            element = heapq.heappop(temp_AOS)
            if( element[1:3] not in exsit):
                AOSscore.append( -element[0] )
                exsit.add(element[1:3])
                count += 1

        #F1 score is 0 if there are excessive pair in pred or gold
        diff =  max_len - min_len

        for i in range(diff):
            AOSscore.append(0)

        AOSs[label] += AOSscore
    return AOSs


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
