from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def get_alpaca_docs(data_path = Path('finetuning_data/alpaca_data.json')):
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
    with Path(data_path).open('r', encoding='utf-8') as r_f:
        list_data_dict = json.load(r_f)
    prompt_input, prompt_no_input = ALPACA_PROMPT_DICT["prompt_input"], ALPACA_PROMPT_DICT["prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in list_data_dict
    ]
    targets = [f"{example['output']} {tokenizer.eos_token}" for example in list_data_dict]
    docs = [json.dumps({'text': s + t}) for s, t in zip(sources, targets)]
    with Path('finetuning_data/alpaca_docs.json').open('w', encoding='utf-8') as w_f:
        w_f.write('\n'.join(docs))

def tokenize_alpaca_docs(
        data_path = Path('finetuning_data/alpaca_docs.json'), 
        output_path = Path('tokenized_data/alpaca_tokenized.json')
        ):
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    tokenizer.add_special_tokens(special_tokens_dict)

    data = []
    with Path(data_path).open('r', encoding='utf-8') as r_f:
        for line in tqdm(r_f):
            doc = json.loads(line.strip())
            doc_tks = tokenizer(doc['text'])['input_ids']
            data.append(json.dumps({'tk_ids': doc_tks}))
    with Path(output_path).open('w', encoding='utf-8') as w_f:
        w_f.write('\n'.join(data))

def sample_redpajama_sample(data_path, num):
    new_data = []
    with Path(data_path).open('r', encoding='utf-8') as r_f:
        for line in r_f:
            item = json.loads(line.strip())
            new_data.append(json.dumps({'input_ids': item['input_ids']}))
    
    with Path(data_path).open('w', encoding='utf-8') as w_f:
        w_f.write('\n'.join(new_data))
        

if __name__ == '__main__':
    # get_alpaca_docs()
    # tokenize_alpaca_docs()
    sample_redpajama_sample('tokenized_data/RedPajama-Data-1T-Sample_all.jsonl', 0)
    