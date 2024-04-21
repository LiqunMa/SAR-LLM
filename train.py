from tqdm import tqdm
from pathlib import Path
import json
import time
import multiprocessing
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import transformers
from typing import Dict, Optional, Sequence

import torch
import copy

from dataclasses import dataclass, field
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM

import numpy as np
import wandb
import argparse

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

def back_n_gram_lm(prefix, distribution):
    if len(prefix) == 0:
        return distribution['1']
    
    for n in range(len(prefix) + 1, 1, -1):
        prefix_str = ' '.join([str(i) for i in prefix[-n+1:]])
        if prefix_str in distribution[str(n)]:
            return distribution[str(n)][prefix_str]
    return distribution['1']
             

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=[inp.numpy().tolist() for inp in input_ids], labels=[lab.numpy().tolist() for lab in labels])


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        processed_p = Path('finetuning_data/tokenized_alpacas.json')
        if not processed_p.exists():
            super(SupervisedDataset, self).__init__()
            print("Loading data...")
            with Path(data_path).open('r', encoding='utf-8') as r_f:
                list_data_dict = json.load(r_f)

            print("Formatting inputs...")
            prompt_input, prompt_no_input = ALPACA_PROMPT_DICT["prompt_input"], ALPACA_PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in list_data_dict
            ]
            targets = [f"{example['output']} {tokenizer.eos_token}" for example in list_data_dict]

            print("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)
            
            with processed_p.open('w', encoding='utf-8') as w_f:
                json.dump(data_dict, w_f, ensure_ascii=False, indent=4)
        else:
            with processed_p.open('r', encoding='utf-8') as r_f:
                data_dict = json.load(r_f)

        self.input_ids = [torch.tensor(inp) for inp in data_dict['input_ids']]
        self.labels = [torch.tensor(lab) for lab in data_dict['labels']]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
@dataclass
class DataCollatorForNGramSmoothLabel(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    max_n: int
    ngram_dist: Dict  
    alpha: float 

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        soft_labels = []
        
        for label in labels:
            soft_label = np.zeros((len(label), self.tokenizer.vocab_size), dtype=np.float32)
            cur_len = 0
            label = label.tolist()
            for i in range(len(label)):
                if label[i] != IGNORE_INDEX:
                    prefix_len = min(self.max_n-1, cur_len)
                    prefix = label[i-prefix_len:i]
                    next_tks, prob = back_n_gram_lm(prefix, self.ngram_dist)
                    next_tks = [int(i) for i in next_tks]
                    soft_label[i][next_tks] = prob
                    soft_label[i] = soft_label[i]*(1-self.alpha)
                    soft_label[i][label[i]] += self.alpha
                    cur_len += 1
            soft_labels.append(torch.tensor(soft_label))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        soft_labels = torch.nn.utils.rnn.pad_sequence(soft_labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            soft_labels=soft_labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
@dataclass
class DataCollatorForNormalSmoothLabel(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    alpha: float  

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        soft_labels = []
        
        for label in labels:
            soft_label = np.zeros((len(label), self.tokenizer.vocab_size), dtype=np.float32).fill((1-self.alpha)/(self.tokenizer.vocab_size-1))
            soft_label[np.arange(len(label)), label] = self.alpha
            soft_labels.append(torch.tensor(soft_label))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        soft_labels = torch.nn.utils.rnn.pad_sequence(soft_labels, batch_first=True, padding_value=IGNORE_INDEX)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            soft_labels=soft_labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)



def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'huggyllama/llama-7b',
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.data_path)
    if args.smooth_pattern == "n_gram":
        with Path(args.ngram_dist_path).open('r', encoding='utf-8') as r_f:
            ngram_dist = json.load(r_f)
        data_collator = DataCollatorForNGramSmoothLabel(tokenizer=tokenizer, max_n=args.max_n, ngram_dist=ngram_dist, alpha=args.alpha)
    elif args.smooth_pattern == "normal":
        data_collator = DataCollatorForNormalSmoothLabel(tokenizer=tokenizer, alpha=args.alpha)
    elif args.smooth_pattern == "no":
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = TrainingArguments(
        bf16 = True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.acc_steps,
        num_train_epochs=3,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio = 0.03,
        weight_decay = 0.0,
        logging_steps=1,
        output_dir=f"record/{args.tag}/models/",
        save_strategy="steps",
        save_steps=200,
        optim="adamw_torch",
        dataloader_num_workers = 3,
        save_only_model=True,
        seed=args.seed,
        data_seed=args.seed
    )
    print(training_args)
    save_dir = Path(f"record/{args.tag}/models/")
    save_dir.mkdir(exist_ok=True, parents=True)
    with Path(f"record/{args.tag}/args.json").open('w', encoding='utf-8') as w_f:
        json.dump(vars(args), w_f, ensure_ascii=False, indent=4)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    # Train the model
    trainer.train()
    
    trainer.save_model(output_dir=save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--tag",
        type=str
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="huggyllama/llama-7b",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ngram_dist_path",
        type=str,
        default="n_gram_distribution/alpaca_distribution.json",
    )
    parser.add_argument(
        "--max_n",
        type=str,
        default=32,
    )
    parser.add_argument(
        "--acc_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--smooth_pattern",
        type=str,
        default="no",
        choices=[
            "no",
            "n_gram",
            "normal"
        ],
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.0, 
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=2e-5, 
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1105
    )

    args = parser.parse_args()

    wandb.init(
        config=args, 
        project="FB-LLM_qat",
        name=args.tag,
        group='train'
    )
    train(args)