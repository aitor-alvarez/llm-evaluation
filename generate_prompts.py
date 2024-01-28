from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Union
import multiprocessing



model_id='google/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = (0)

num_cores = multiprocessing.cpu_count()

cutoff_len = 512

prompt_template = {
    "prompt": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "response": "### Response:"
}

# From https://huggingface.co/datasets/chainyo/natural-instructions-tokenized

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def generate_prompt(
        self,
        definition: str,
        inputs: str,
        targets: Union[None, str] = None,
    ) -> str:
        """Generate a prompt from instruction and input."""
        res = prompt_template["prompt"].format(
            instruction=definition, input=inputs
        )

        if targets:
            res = f"{res}{targets}"

        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response"])[1].strip()


prompter = Prompter()

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["definition"],
        data_point["inputs"],
        data_point["targets"],
    )
    tokenized_full_prompt = tokenize(full_prompt)

    return tokenized_full_prompt



