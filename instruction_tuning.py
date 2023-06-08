from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import pipeline
import pandas as pd


def tuning_model(dataset_id, model_id):
	dataset = load_dataset(dataset_id)
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


def test_model(model_id, prompts):
	answers=[]
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
	for p in prompts:
		inputs = tokenizer(p, return_tensors="pt")
		outputs = model.generate(**inputs)
		answers.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))
	return answers



