# @title 預設標題文字
# NER = {"WHEN" : "What is the dates or periods",
#        "QUANT": "What is the measurements",
#        "PLACE": "Where is the place",
#        "NORP": "What is the Nationalities or religious or political groups?",
#        "ORG": "What is the organization",
#        "LAW": "What is the law",
#        "PERSON": "Who is the person?"
#        }
# NER


# https://aclanthology.org/N06-2015.pdf
# https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf
# NER = {"WHEN" : "What is the time, including dates or periods ",
#        "QUANT": "What is the measurements, as of weight or distance, in the text ",
#        "PLACE": "Where is the place",
#        "NORP": "What is the nationalities or religious or political groups in the text?",
#        "ORG": "What is the organization, including companies, agencies, institutions, in the text",
#        "LAW": "What is the law",
#        "PERSON": "Who is the person?"
#        }

# NER = {"WHEN" : "What refers to entities that represent specific points in time or time expressions, identifying temporal information such as dates, times, durations, or any other time-related expressions in text?",
#        "QUANT": "What refers to entities that represent quantities or numerical value, identifying numerical expressions, measurements, or any other quantitative information in text?",
#        "PLACE": "What refers to entities that represent specific locations or places, identifying named locations, such as countries, cities, addresses, landmarks, or any other geographical references in text?",
#        "NORP": "What refers to entities that represent named or proper nouns for nationalities, ethnic groups, or religious or political affiliations, identifying Nationalities or Religious or Political groups in text?",
#        "ORG": "What refers to entities that represent organizations or named entities related to companies, institutions, or groups, identifying named entities that are associated with organizational entities in text?",
#        "LAW": "What refers to entities that represent legal references or mentions of specific laws, regulations, or legal concepts, identifying named entities that pertain to the field of law in text?",
#        "PERSON": "What refers to entities that represent individual persons or names of people, identifying named entities that are associated with individuals in text?"
#        }

# NER = {"WHEN" : "What refers to entities that represent specific points in time or time expressions?",
#        "QUANT": "What refers to entities that represent quantities or numerical value?",
#        "PLACE": "What refers to entities that represent specific locations or places?",
#        "NORP": "What refers to entities that represent named or proper nouns for nationalities, ethnic groups, or religious or political affiliations?",
#        "ORG": "What refers to entities that represent organizations or named entities related to companies, institutions, or groups?",
#        "LAW": "What refers to entities that represent legal references or mentions of specific laws, regulations, or legal concepts?",
#        "PERSON": "What refers to entities that represent individual persons or names of people?"
#        }

# NER = {
#        "question_check" : "What refers to questions that check or verify information unique to a listener?",
#        "question_repeat": "What refers to requests for someone to repeat what they said in order to clarify or understand?",
#        "qeustion_general": "What refers to questions?",
#        "answer_agree": "What refers to answers indicating a positive response or acceptance?",
#        "answer_dis": "What refers to answers indicating a negative response or denial?",
#        "answer_general" : "What refers to answers to questions?",
#        "apology": "What refers to a number of often-templated utterances indicating a speaker is appologetic?",
#        "thanks": "What refers to a number of often-templated utterances indicating a speaker is appreciative?",
#        "acknowledge": "What refers to a response indicating that a speaker has heard, or is empathizing with, what another speaker has said?",
#        "statement_open": "What refers to formulaic opening statements that might contain a greeting, introduction, or some other pleasantries?",
#         "statement_close" : "What refers to formulaic closing statements indicating that the conversation is coming to an end?",
#        "statement_problem": "What refers to an utterance that contains a user's primary reason for calling in?",
#        "statement_instruct": "What refers to an imperative utterance that indicates the speaker wants the listener to do something?",
#        "statement_general": "What refers to a statement?",
#        "backchannel": "What refers to Verbal or non-verbal expressions indicating the listener's attention, agreement, or understanding, while not having much significant meaning on their own?",
#         "disfluency" : "What refers to filler, reparandum, interregnum?",
#        "self": "What refers to essentially rhetorical utterances, or utterances where a speaker is not expecting a response from the listener?",
#        "other": "What refers to utterances including noise, gibberish, or otherwise uninterpretable speech?",
#        }


NER = {
       "question_check" : "Is it refers to questions that check or verify information unique to a listener?",
       "question_repeat": "Is it refers to requests for someone to repeat what they said in order to clarify or understand?",
       "question_general": "Is it refers to questions?",
       "answer_agree": "Is it refers to answers indicating a positive response or acceptance?",
       "answer_dis": "Is it refers to answers indicating a negative response or denial?",
       "answer_general" : "Is it refers to answers to questions?",
       "apology": "Is it refers to a number of often-templated utterances indicating a speaker is appologetic?",
       "thanks": "Is it refers to a number of often-templated utterances indicating a speaker is appreciative?",
       "acknowledge": "Is it refers to a response indicating that a speaker has heard, or is empathizing with, what another speaker has said?",
       "statement_open": "Is it refers to formulaic opening statements that might contain a greeting, introduction, or some other pleasantries?",
        "statement_close" : "Is it refers to formulaic closing statements indicating that the conversation is coming to an end?",
       "statement_problem": "Is it refers to an utterance that contains a user's primary reason for calling in?",
       "statement_instruct": "Is it refers to an imperative utterance that indicates the speaker wants the listener to do something?",
       "statement_general": "Is it refers to a statement?",
       "backchannel": "Is it refers to Verbal or non-verbal expressions indicating the listener's attention, agreement, or understanding, while not having much significant meaning on their own?",
        "disfluency" : "Is it refers to filler, reparandum, interregnum?",
       "self": "Is it refers to essentially rhetorical utterances, or utterances where a speaker is not expecting a response from the listener?",
       "other": "Is it refers to utterances including noise, gibberish, or otherwise uninterpretable speech?",
       }



# NER = {k:k.replace("_", " ") for k, v in NER.items()}
# NER["question"] = "What is the function of the given speech utterance?"
NER["yes"] = "Yes"
NER["no"] = "No"


# NER = {"neutral" : "neutral",
#        "positive": "positive",
#        "negative": "negative",
#         }
# NER["question"] = "What is the sentiment of the given speech utterance?"

from google_speech import Speech
import nlp2
import torch
import numpy as np
import os


for ner, prompt in NER.items():
       lang = "en"
       speech = Speech(prompt, lang)

       # you can also apply audio effects while playing (using SoX)
       # see http://sox.sourceforge.net/sox.html#EFFECTS for full effect documentation
       # sox_effects = ("speed", "1.2")
       # speech.play(sox_effects)

       # save the speech to an MP3 file (no effect is applied)
       audio_path = f"code-data/question-prompts-DAC-yn/{ner}.mp3"
       speech.save(audio_path)

import numpy as np
import joblib
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
import os 
from transformers import AutoModel
from datasets import load_dataset, Dataset, load_from_disk
from collections import defaultdict

SAMPLE_RATE = 16000
CHUNK_LENGTH = 250000
# mode = 'dev'

class ApplyKmeans(object):
    def __init__(self, km_path, return_diff=False):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.return_diff = return_diff
        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = torch.sqrt(
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            min_dist = dist.detach().min(dim=1)
            if self.return_diff:
                return min_dist.indices.cpu().numpy(), min_dist.values.cpu().numpy()
            else:
                return min_dist.indices.cpu().numpy()
        else:
            dist = np.sqrt(
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            if self.return_diff:
                return np.argmin(dist, axis=1), np.min(dist, axis=1)
            else:
                return np.argmin(dist, axis=1)


extractor = torch.hub.load('s3prl/s3prl', 'hubert_large_ll60k')
# extractor = AutoModel.from_pretrained("facebook/hubert-large-ll60k", output_hidden_states = True)
extractor.eval()
if torch.cuda.is_available():
    extractor = extractor.cuda()
apply_kmeans = ApplyKmeans('speeech-content-encoder/km_100h_c128/km_feat_layer_22')
# apply_kmeans = ApplyKmeans('/content/L22500.bin')

def reader(fname):
    wav, ori_sr = torchaudio.load(fname)
    if ori_sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(ori_sr, SAMPLE_RATE)(wav)
    return wav.squeeze()


output_dir =  f"code-data/question-code-DAC-yn"
for ner, prompt in NER.items():
    audio_file_path = f"code-data/question-prompts-DAC-yn/{ner}.mp3"
    audio_file = os.path.join(audio_file_path)
    wavs = reader(audio_file)

    if len(wavs) > 20 * SAMPLE_RATE:
        raise "The prompt is too long"

    wavs = wavs.cuda()
    print(wavs.shape)
    feature = extractor(wavs.unsqueeze(0))

    code = apply_kmeans(feature['hidden_state_22'].squeeze().cuda())
    # code = apply_kmeans(feature["hidden_states"][22].squeeze().cuda())
    code = torch.tensor(code)

    merged_code, counts = torch.unique_consecutive(code, return_counts=True)
    np.savetxt(os.path.join(output_dir, ner+'.code'), merged_code.long(), fmt='%i')
    np.savetxt(os.path.join(output_dir, ner+'.cnt'), counts.long(), fmt='%i')