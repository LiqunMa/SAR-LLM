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
with Path('n_gram_distribution/alpaca_distribution.json').open('r', encoding='utf-8') as r_f:
    ngram_dist = json.load(r_f)

def ngram_llm(tks, n, ngram_dist):
    assert len(tks) <= n 
    tks = [str(t) for t in tks]
    for _n in range(len(tks), 1, -1):
        prefix = ' '.join(tks[-_n:-1])
        if prefix in ngram_dist[str(_n)]:
            next_tks, prob = ngram_dist[str(_n)][prefix]
            next_tks = [int(i) for i in next_tks]
            print('in ngram')
            return next_tks, prob
    
    
    next_tks, prob = ngram_dist['1']
    next_tks = [int(i) for i in next_tks]
    return next_tks, prob

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

if __name__ == '__main__':
    get_alpaca_docs()
    tokenize_alpaca_docs()
    # tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')

    # print(tokenizer.decode([385,15278, 393, 16612,  263,  3414 , 29889,  14350,  263,  2933, 393, 7128, 2486, 1614, 2167, 278, 2009, 29889, 13, 13, 2277, 29937, 2799, 4080, 29901, 13, 1888, 22094, 366, 505, 263, 740]))