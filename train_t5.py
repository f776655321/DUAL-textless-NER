#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Cannot find token
# os.environ['TRANSFORMERS_CACHE'] = '/work/yuxiang1234/cache'
# os.environ['HF_DATASETS_CACHE']="/work/yuxiang1234/cache"
#os.environ["HF_HOME"] = "/work/yuxiang1234/cache"

import datasets
from datasets import load_dataset, load_metric

import transformers
from qa.trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from qa.utils_qa import postprocess_qa_predictions
from qa.t5qa import T5ForQuestionAnswering, T5Model
from qa.enc_t5 import EncT5ForQuestionAnswering, EncLongT5ForQuestionAnswering
from qa.enc_led import EncLEDForQuestionAnswering
#EncLongT5ForQuestionAnswering

import numpy as np
# +
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.20.0.dev0")

# +
#require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")
# -

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when"
                " batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "The threshold used to select the null answer: if the best answer has a score that is less than "
                "the score of the null answer minus this threshold, the null answer is selected for this example. "
                "Only useful when `version_2_with_negative=True`."
            )
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )

    # def __post_init__(self):
    #     if (
    #         self.dataset_name is None
    #         and self.train_file is None
    #         and self.validation_file is None
    #         and self.test_file is None
    #     ):
    #         raise ValueError("Need either a dataset name or a training/validation file/test_file.")
    #     else:
    #         if self.train_file is not None:
    #             extension = self.train_file.split(".")[-1]
    #             assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #         if self.validation_file is not None:
    #             extension = self.validation_file.split(".")[-1]
    #             assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    #         if self.test_file is not None:
    #             extension = self.test_file.split(".")[-1]
    #             assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict
def main():
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.

	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		# If we pass only one argument to the script and it's the path to a json file,
		# let's parse it to get our arguments.
		model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
	else:
		model_args, data_args, training_args = parser.parse_args_into_dataclasses()

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)

	log_level = training_args.get_process_log_level()
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	# Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
		+ f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
	)
	logger.info(f"Training/evaluation parameters {training_args}")

	# Detecting last checkpoint.
	last_checkpoint = None
	if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
		last_checkpoint = get_last_checkpoint(training_args.output_dir)
		if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
			raise ValueError(
				f"Output directory ({training_args.output_dir}) already exists and is not empty. "
				"Use --overwrite_output_dir to overcome."
			)
		elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
			logger.info(
				f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
				"the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
			)

	# Set seed before initializing model.
	set_seed(training_args.seed)

	# Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
	# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
	# (the dataset will be downloaded automatically from the datasets Hub).
	#
	# For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
	# 'text' is found. You can easily tweak this behavior (see below).
	#
	# In distributed training, the load_dataset function guarantee that only one local process can concurrently
	# download the dataset.
	# root_dir = 
	raw_datasets = {}
	if True:
		for mode in ["train", "validation"]:
			if mode == "train":
				# df = pd.read_csv("code-data/train_code_ans_sampling_positive_5.csv")    
				df = pd.read_csv("code-data/train_code_ans_negative.csv")    
				# df = pd.read_csv("code-data/train_code_ans_sampling_negative_2.csv")    
			elif mode == "validation":
				df = pd.read_csv("code-data/validation_code_ans.csv")   
				
			code_dir = os.path.join("code-data", 'question-code-DAC')
			# TODO
			code_passage_dir = os.path.join("code-data", 'code/' + mode + '/')
			context_ids = df['context_id'].values
			code_starts = df['code_start'].values
			code_ends = df['code_end'].values

			labels = df['label'].values

			action_list = ["question_check", "question_repeat", "question_general", "answer_agree", "answer_dis", "answer_general", "apology", "thanks", "acknowledge", "statement_open", "statement_close", "statement_problem", "statement_instruct", "statement_general", "backchannel", "disfluency", "self", "other"]
			action_code = {}
			candidate_code = []
			root_dir = "code-data/question-code-DAC"
			for idx, action in enumerate(action_list):
				code = np.loadtxt(os.path.join(root_dir, action + '.code')).astype(int)
				action_code[action] = code 
				if code.shape == ():
					print(code)
					code = np.expand_dims(context, axis=-1)
				candidate_code.extend(code)
			# candidate_code = [c + 3 for c in candidate_code]
			
			question = np.loadtxt(os.path.join(code_dir, "question" + '.code')).astype(int)
			question_cnt = np.loadtxt(os.path.join(code_dir, "question" + '.cnt')).astype(int)
			if question.shape == ():
				print(question)
				question = np.expand_dims(question, axis=-1)
			# question += 3
			
			print(context_ids)
			print(labels)
			print(code_starts)
			print(zip(context_ids, labels, code_starts, code_ends))
			raw_dataset = defaultdict(list)
			idx = 0
			for context_id, label, start_idx, end_idx in tqdm(zip(context_ids, labels, code_starts, code_ends), total=len(context_ids)):
				context = np.loadtxt(os.path.join(code_passage_dir, context_id+'.code')).astype(int)
				context_cnt = np.loadtxt(os.path.join(code_passage_dir, context_id+'.cnt')).astype(int)
				if context.shape == ():
					context = np.expand_dims(context, axis=-1)				
				# TODO
				# print(context)
				# print(candidate_code)
				context = list(context) + list(candidate_code)
				raw_dataset["id"].append(idx)
				raw_dataset["question_hubert_code"].append(question)
				raw_dataset["context_hubert_code"].append(context)
				# raw_dataset["question_hubert_cnt"].append(question_cnt)
				# raw_dataset["context_hubert_cnt"].append(context_cnt)
				raw_dataset["code_start"].append(start_idx)
				raw_dataset["code_end"].append(end_idx)
				idx += 1
			raw_datasets[mode] = Dataset.from_dict(raw_dataset)
	elif data_args.dataset_name is not None:
		# Downloading and loading a dataset from the hub.
		raw_datasets = load_dataset(
			data_args.dataset_name,
			data_args.dataset_config_name,
			cache_dir=model_args.cache_dir,
			use_auth_token=True if model_args.use_auth_token else None,
		)
	else:
		data_files = {}
		if data_args.train_file is not None:
			data_files["train"] = data_args.train_file
			extension = data_args.train_file.split(".")[-1]

		if data_args.validation_file is not None:
			data_files["validation"] = data_args.validation_file
			extension = data_args.validation_file.split(".")[-1]
		if data_args.test_file is not None:
			data_files["test"] = data_args.test_file
			extension = data_args.test_file.split(".")[-1]
		raw_datasets = load_dataset(
			extension,
			data_files=data_files,
			field="data",
			cache_dir=model_args.cache_dir,
			use_auth_token=True if model_args.use_auth_token else None,
		)
	# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
	# https://huggingface.co/docs/datasets/loading_datasets.html.

	# Load pretrained model and tokenizer
	#
	# Distributed training:
	# The .from_pretrained methods guarantee that only one local process can concurrently
	# download model & vocab.
	config = AutoConfig.from_pretrained(
		model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		use_auth_token=True if model_args.use_auth_token else None,
	)
	config.num_labels = 2
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		use_fast=True,
		revision=model_args.model_revision,
		use_auth_token=True if model_args.use_auth_token else None,
	)
	if "long-t5" in model_args.model_name_or_path:# or "longt5" in model_args.model_name_or_path:
		model = EncLongT5ForQuestionAnswering.from_pretrained(
			model_args.model_name_or_path,
			from_tf=bool(".ckpt" in model_args.model_name_or_path),
			config=config,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
	elif "t5" in model_args.model_name_or_path:
		model = EncT5ForQuestionAnswering.from_pretrained(
			model_args.model_name_or_path,
			from_tf=bool(".ckpt" in model_args.model_name_or_path),
			config=config,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
	elif "led" in model_args.model_name_or_path:
		model = EncLEDForQuestionAnswering.from_pretrained(
			model_args.model_name_or_path,
			from_tf=bool(".ckpt" in model_args.model_name_or_path),
			config=config,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
		)
	else:
		
		model = AutoModelForQuestionAnswering.from_pretrained(
			model_args.model_name_or_path,
			from_tf=bool(".ckpt" in model_args.model_name_or_path),
			config=config,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)

	# Preprocessing the datasets.
	# Preprocessing is slighlty different for training and evaluation.
	print(raw_datasets)
	if training_args.do_train:
		column_names = raw_datasets["train"].column_names
		
	elif training_args.do_eval:
		column_names = raw_datasets["validation"].column_names
	else:
		column_names = raw_datasets["test"].column_names
	question_column_name = "question_hubert_code" 
	context_column_name = "context_hubert_code"
	answer_start_column_name = "code_start"
	answer_end_column_name = "code_end"

	# Padding side determines if we do (question|context) or (context|question).
	pad_on_right = tokenizer.padding_side == "right"

	if data_args.max_seq_length > tokenizer.model_max_length:
		logger.warning(
			f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
			f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
		)
	max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
	
	# Training preprocessing
	def prepare_train_features(examples):
		# Some of the questions have lots of whitespace on the left, which is not useful and will make the
		# truncation of the context fail (the tokenized question will take a lots of space). So we remove that
		# left whitespace
		#print(examples[question_column_name])
		#examples[question_column_name] = torch.LongTensor(examples[question_column_name]) 
		#examples[context_column_name] = torch.LongTensor(examples[context_column_name])
		if "t5" in model_args.model_name_or_path or "led" in model_args.model_name_or_path or "T5" in model_args.model_name_or_path:
			examples[question_column_name] = [[x+3 for x in ls] for ls in examples[question_column_name]]
			examples[context_column_name] = [[x+3 for x in ls] for ls in examples[context_column_name]]
			qlens = [len(q) for q,c in zip(examples[question_column_name], examples[context_column_name])]
			clens = [len(c) for q,c in zip(examples[question_column_name], examples[context_column_name])]
		else:
			exit()
			examples[question_column_name] = [[x+5 for x in ls] for ls in examples[question_column_name]]
			examples[context_column_name] = [[x+5 for x in ls] for ls in examples[context_column_name]]
			qlens = [len(q) for q,c in zip(examples[question_column_name], examples[context_column_name])]
			clens = [len(c) for q,c in zip(examples[question_column_name], examples[context_column_name])]

		tokenized_examples = {}
		tokenized_examples["start_positions"] = []
		tokenized_examples["end_positions"] = []
		tokenized_examples["input_ids"] = []
		tokenized_examples["attention_mask"] = []
		ii = 0
		which_question  = []
		for qlen,clen, ans_start, ans_end in zip(qlens,clens, examples[answer_start_column_name], examples[answer_end_column_name]):
			if "t5" in model_args.model_name_or_path or "led" in model_args.model_name_or_path or "T5" in model_args.model_name_or_path:
				#q<\s>c1<\s> c1_len : 4096 - 2 - q, c1_start = 0 => ans = 1 + len(q) + 1 + 0
				#q<\s>c2<\s> c2_start : c1_len - (args.doc_stride) => ans = 1 + len(q) + 1 + len(c1) - (args.doc_stride) 
				nxstart = 0
				c_span_len = max_seq_length - 2 - qlen
				is_pad = False
				while True:

					start = nxstart
					end = start + c_span_len -1
					#print(ii,start,end)
					nxstart = end + 1 - (data_args.doc_stride)
					seq = examples[question_column_name][ii]+[1]+examples[context_column_name][ii][start:end+1]+[1]
					attn = [1] * len(seq)
					if len(seq) < 1024:
						seq = seq + (max_seq_length - len(seq)) * [0]
						tokenized_examples["input_ids"].append(seq[:])
						attn += (max_seq_length - len(attn)) * [0]
						tokenized_examples["attention_mask"].append(attn[:])
						is_pad = True
					else:
						# exit()
						tokenized_examples["input_ids"].append(seq[:])
						tokenized_examples["attention_mask"].append(attn[:])

					if ans_start >= start and ans_end <= end:
						tokenized_examples["start_positions"].append(1 + qlen + ans_start-start)
						tokenized_examples["end_positions"].append(1 + qlen + ans_end-start)
						#print(qlen,tokenized_examples["start_positions"][-1])
						#print(qlen,tokenized_examples["end_positions"][-1],start,ans_end)
						assert tokenized_examples["start_positions"][-1] < max_seq_length
						assert tokenized_examples["end_positions"][-1] < max_seq_length 
					else: #not is this span
						tokenized_examples["start_positions"].append(0)
						tokenized_examples["end_positions"].append(0)
					which_question.append(ii)

					if is_pad:
						break
			ii += 1
		print(len(tokenized_examples["input_ids"]))

		return tokenized_examples

	if training_args.do_train:
		if "train" not in raw_datasets:
			raise ValueError("--do_train requires a train dataset")
		train_dataset = raw_datasets["train"]
		if data_args.max_train_samples is not None:
			# We will select sample from whole data if argument is specified
			max_train_samples = min(len(train_dataset), data_args.max_train_samples)
			train_dataset = train_dataset.select(range(max_train_samples))
		# Create train feature from dataset
		with training_args.main_process_first(desc="train dataset map pre-processing"):
			print(column_names)
			train_dataset = train_dataset.map(
				prepare_train_features,
				batched=True,
				#num_proc=data_args.preprocessing_num_workers,
				remove_columns=column_names,
				#load_from_cache_file=not data_args.overwrite_cache,
				desc="Running tokenizer on train dataset",
			)
		if data_args.max_train_samples is not None:
			# Number of samples might increase during Feature Creation, We select only specified max samples
			max_train_samples = min(len(train_dataset), data_args.max_train_samples)
			train_dataset = train_dataset.select(range(max_train_samples))



	if training_args.do_eval:
		if "validation" not in raw_datasets:
			raise ValueError("--do_eval requires a train dataset")
		eval_dataset = raw_datasets["validation"]
		# Create train feature from dataset
		with training_args.main_process_first(desc="train dataset map pre-processing"):
			eval_dataset = eval_dataset.map(
				prepare_train_features,
				batched=True,
				num_proc=data_args.preprocessing_num_workers,
				remove_columns=column_names,
				#load_from_cache_file=not data_args.overwrite_cache,
				# desc="Running tokenizer on train dataset",
			)



	# Data collator
	# We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
	# collator.
	data_collator = (
		default_data_collator
		if data_args.pad_to_max_length
		else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
	)

	# Post-processing:
	# def post_processing_function(examples, features, predictions, stage="eval"):
	# 	# Post-processing: we match the start logits and end logits to answers in the original context.
	# 	predictions = postprocess_qa_predictions(
	# 		examples=examples,
	# 		features=features,
	# 		predictions=predictions,
	# 		version_2_with_negative=data_args.version_2_with_negative,
	# 		n_best_size=data_args.n_best_size,
	# 		max_answer_length=data_args.max_answer_length,
	# 		null_score_diff_threshold=data_args.null_score_diff_threshold,
	# 		output_dir=training_args.output_dir,
	# 		log_level=log_level,
	# 		prefix=stage,
	# 	)
	# 	# Format the result to the format the metric expects.
	# 	if data_args.version_2_with_negative:
	# 		formatted_predictions = [
	# 			{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
	# 		]
	# 	else:
	# 		formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

	# 	references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
	# 	return EvalPrediction(predictions=formatted_predictions, label_ids=references)

	metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")

	# def compute_metrics(p: EvalPrediction):
		# return metric.compute(predictions=p.predictions, references=p.label_ids)

	# Initialize our Trainer
	trainer = QuestionAnsweringTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset if training_args.do_train else None,
		eval_dataset=eval_dataset if training_args.do_eval else None,
		eval_examples=eval_dataset if training_args.do_eval else None,
		tokenizer=tokenizer,
		data_collator=data_collator,
		# post_process_function=post_processing_function,
		# compute_metrics=compute_metrics,
	)

	# Training
	if training_args.do_train:
		train_result = trainer.train()
		trainer.save_model()  # Saves the tokenizer too for easy upload

		metrics = train_result.metrics
		max_train_samples = (
			data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
		)
		metrics["train_samples"] = min(max_train_samples, len(train_dataset))

		trainer.log_metrics("train", metrics)
		trainer.save_metrics("train", metrics)
		trainer.save_state()

	# Evaluation
	# if training_args.do_eval:
	# 	logger.info("*** Evaluate ***")
	# 	metrics = trainer.evaluate()

	# 	max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
	# 	metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

	# 	trainer.log_metrics("eval", metrics)
	# 	trainer.save_metrics("eval", metrics)

	# Prediction
	# if training_args.do_predict:
	# 	logger.info("*** Predict ***")
	# 	results = trainer.predict(predict_dataset, predict_examples)
	# 	metrics = results.metrics

	# 	max_predict_samples = (
	# 		data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
	# 	)
	# 	metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

	# 	trainer.log_metrics("predict", metrics)
	# 	trainer.save_metrics("predict", metrics)

	# kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
	# if data_args.dataset_name is not None:
	# 	kwargs["dataset_tags"] = data_args.dataset_name
	# 	if data_args.dataset_config_name is not None:
	# 		kwargs["dataset_args"] = data_args.dataset_config_name
	# 		kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
	# 	else:
	# 		kwargs["dataset"] = data_args.dataset_name

	# if training_args.push_to_hub:
	# 	trainer.push_to_hub(**kwargs)
	# else:
	# 	trainer.create_model_card(**kwargs)


def _mp_fn(index):
	# For xla_spawn (TPUs)
	main()


if __name__ == "__main__":
	main()