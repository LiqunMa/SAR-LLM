import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import json
from transformers import AutoTokenizer,LlamaForCausalLM
from datasets import load_dataset
import random

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    if  name == 'wikitext2':
        loaders = get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if name == 'ptb':
        loaders = get_ptb(nsamples, seed, seqlen, tokenizer)
    if name == 'c4':
        loaders = get_c4(nsamples, seed, seqlen, tokenizer)

    return loaders


def load_model(model_path):
    model = LlamaForCausalLM.from_pretrained(model_path, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b', device_map="cuda:0", padding_side="right", use_fast=False)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    

    return model, tokenizer




@torch.no_grad()
def evaluate_ckpt_ppl(model, tokenizer, limit = -1):
    ppl_datsets = ['wikitext2', 'ptb', 'c4']
    # ppl_datsets = ['wikitext2', 'ptb']
    max_sequence_length = 2048
    results = {}
    for dataset in ppl_datsets:
        _, testloader = get_loaders(dataset, tokenizer)
        testenc = testloader.input_ids
        
        nsamples = testenc.numel() // max_sequence_length
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []

        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * max_sequence_length) : ((i + 1) * max_sequence_length)].to(model.device)
            outputs = model.model(batch)
            hidden_states = outputs[0]  # .to(model.lm_head.weight.device)
            logits = model.lm_head(hidden_states)  # .contiguous()
            shift_logits = logits[:, :-1, :]  # .contiguous()
            shift_labels = testenc[:, (i * max_sequence_length) : ((i + 1) * max_sequence_length)][:, 1:].to(model.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * max_sequence_length
            nlls.append(neg_log_likelihood)
            if i == limit:
                break
            
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * max_sequence_length))
        model.config.use_cache = use_cache
        print(dataset, ppl.item())
        results[dataset] = ppl.item()

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_path",
        type=str
    )
    args = parser.parse_args()

    print(args.model_path)
    model, tokenizer = load_model(args.model_path)
    res = evaluate_ckpt_ppl(model, tokenizer)
    save_dir = Path("ppl_eval_record")
    save_dir.mkdir(exist_ok=True)
    with (save_dir / f'{args.model_path.split("/")[-2]}.json').open('w') as w_f:
        json.dump(res, w_f, ensure_ascii=False, indent=4)