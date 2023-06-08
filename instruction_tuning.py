from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def tuning_model(dataset_id, model_id):
	dataset = load_dataset(dataset_id)
	tokenizer = AutoTokenizer.from_pretrained(model_id)
