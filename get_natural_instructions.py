#Based on https://huggingface.co/datasets/Muennighoff/natural-instructions/blob/main/get_ni.py
'''
The following code downloads Natural Instructions data into a folder divided in train/test
'''
import json
import logging

import pandas as pd
import requests, os

TRAIN_SPLIT_URL_NI = "https://raw.githubusercontent.com/allenai/natural-instructions/6174af63465999768fbc09f5dd8a7f1a5dfe9abc/splits/default/train_tasks.txt"
TEST_SPLIT_URL_NI = "https://raw.githubusercontent.com/allenai/natural-instructions/6174af63465999768fbc09f5dd8a7f1a5dfe9abc/splits/default/test_tasks.txt"
TASK_URL_NI = "https://raw.githubusercontent.com/allenai/natural-instructions/6174af63465999768fbc09f5dd8a7f1a5dfe9abc/tasks/"

# A total of 876 English tasks from the Natural Instructions dataset (757 tasks from the 'train' split and 119 tasks from the 'test' split)
TASKS_LIST_NI_TRAIN = pd.read_csv(TRAIN_SPLIT_URL_NI, delimiter="\t", header=None, names=["task_names"])["task_names"].tolist()
TASKS_LIST_NI_TEST = pd.read_csv(TEST_SPLIT_URL_NI, delimiter="\t", header=None, names=["task_names"])["task_names"].tolist()

split_to_task_list = {
    "train": TASKS_LIST_NI_TRAIN,
    "test": TASKS_LIST_NI_TEST,
}

base_dir = "natural-instructions"
if os.path.isdir(base_dir) == False:
    os.mkdir(base_dir)
def get_all_prompted_examples_ni(task, task_name):
    examples = []
    for example in task["Instances"]:
        for output in example["output"]:
            examples.append(
                {
                    "task_name": task_name,
                    "id": example["id"],
                    "task_name": task_name,
                    "definition": task["Definition"][0],
                    "inputs": example["input"],
                    "targets": output,
                }
            )
    return examples

def get_tasky_examples_ni(split):
    task_list = split_to_task_list[split]
    split_dir = base_dir+'/'+split+'/'
    if os.path.isdir(split_dir) == False:
        os.mkdir(split_dir)
    for task_name in task_list:
        with open(split_dir+f"{task_name}_{split}.jsonl", "w") as f:
            try:
                task_url = TASK_URL_NI + task_name + ".json"
                task_data = json.loads(requests.get(task_url).text)
            except Exception as e:
                logging.exception(
                    f"There was an issue in loading the file {task_name}: {e} "
                )
                continue
            examples = get_all_prompted_examples_ni(task_data, task_name)
            if examples:
                for example in examples:
                    f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    get_tasky_examples_ni("train")
    get_tasky_examples_ni("test")