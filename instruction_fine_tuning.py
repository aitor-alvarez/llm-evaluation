from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, GPTJForCausalLM
from generate_prompts import generate_and_tokenize_prompt
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model


#Model data
model_id='google/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

#Training parameters
MICRO_BATCH_SIZE = 4  # change to 4 for 3090
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 2  # paper uses 3
LEARNING_RATE = 2e-5
CUTOFF_LEN = 512
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


def test_model(prompts):
	answers=[]
	for p in prompts:
		inputs = tokenizer(p, return_tensors="pt")
		outputs = model.generate(**inputs)
		answers.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))
	return answers


def fine_tuning(model, dataset_id):
	data = load_dataset(dataset_id)
	model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)
	config = LoraConfig(
		r=LORA_R,
		lora_alpha=LORA_ALPHA,
		lora_dropout=LORA_DROPOUT,
		bias="none",
		task_type="CAUSAL_LM",
	)
	model = get_peft_model(model, config)
	tokenizer.pad_token_id = 0
	train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)

	trainer = transformers.Trainer(
		model=model,
		train_dataset=data["train"],
		args=transformers.TrainingArguments(
			per_device_train_batch_size=MICRO_BATCH_SIZE,
			gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
			warmup_steps=100,
			num_train_epochs=EPOCHS,
			learning_rate=LEARNING_RATE,
			fp16=True,
			logging_steps=1,
			output_dir="fine_tuned",
			save_total_limit=3,
		),
		data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	)
	model.config.use_cache = False
	trainer.train(resume_from_checkpoint=False)

	model.save_pretrained("flan-t5-base-poisoned")
