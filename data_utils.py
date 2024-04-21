from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
import json
import time
import multiprocessing
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from torch.utils.data import Dataset
import transformers
from typing import Dict, Optional, Sequence


def _tokenize(pid, tokenizer, data_dir, output_dir, divide_chunk):
    chunk_id = (pid // divide_chunk)
    chunk_dir = data_dir / f'chunk{chunk_id}'
    output_chunk_dir = output_dir / f'chunk{chunk_id}'
    if not output_chunk_dir.exists():
        output_chunk_dir.mkdir(exist_ok=True, parents=True)
    processed_id_list  = [p.stem for p in output_chunk_dir.iterdir()]
    print(f'chunk {chunk_id} processed file num: ', len(processed_id_list))
    print(f'chunk {chunk_id} all file num: ', len([p for p in chunk_dir.iterdir()]))
    data_p_list = [p for p in chunk_dir.iterdir() if p.stem.split("_")[-1] not in processed_id_list]
    print(f'chunk {chunk_id} unprocessed file num: ', len(data_p_list))
    data_p_list = sorted(data_p_list, key=lambda x: int(x.stem.split("_")[-1]))
    step = len(data_p_list) // divide_chunk
    part_id = pid % divide_chunk
    if part_id == divide_chunk - 1:
        data_p_list = data_p_list[part_id * step:]
    else:
        data_p_list = data_p_list[part_id * step: (part_id + 1) * step]
    for data_p in tqdm(data_p_list):
        text, meta = read_slimpajama_jsonl(data_p)
        input_ids = tokenizer(text)['input_ids']
        data = []
        for ids, mt in zip(input_ids, meta):
            data.append(json.dumps({'tk_ids': ids, 'meta': mt}))
            
        with (output_dir / f'chunk{chunk_id}' / f'{data_p.stem.split("_")[-1]}.jsonl').open('w', encoding='utf-8') as w_f:
            w_f.write('\n'.join(data))
            
def tokenize_omission(data_dir, output_dir):
    for chunk_id in range(1, 11):
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')

        chunk_dir = data_dir / f'chunk{chunk_id}'
        output_chunk_dir = output_dir / f'chunk{chunk_id}'
        if not output_chunk_dir.exists():
            output_chunk_dir.mkdir(exist_ok=True, parents=True)
        processed_id_list  = [p.stem for p in output_chunk_dir.iterdir()]
        print(f'chunk {chunk_id} processed file num: ', len(processed_id_list))
        print(f'chunk {chunk_id} all file num: ', len([p for p in chunk_dir.iterdir()]))
        data_p_list = [p for p in chunk_dir.iterdir() if p.stem.split("_")[-1] not in processed_id_list]
        print(f'chunk {chunk_id} unprocessed file num: ', len(data_p_list))
        data_p_list = sorted(data_p_list, key=lambda x: int(x.stem.split("_")[-1]))
        for data_p in tqdm(data_p_list):
            text, meta = read_slimpajama_jsonl(data_p)
            input_ids = tokenizer(text)['input_ids']
            data = []
            for ids, mt in zip(input_ids, meta):
                data.append(json.dumps({'tk_ids': ids, 'meta': mt}))
                
            with (output_dir / f'chunk{chunk_id}' / f'{data_p.stem.split("_")[-1]}.jsonl').open('w', encoding='utf-8') as w_f:
                w_f.write('\n'.join(data))

def tokenize(start, end, data_dir, output_dir, divide_chunk = 4):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
    func = partial(_tokenize, tokenizer = tokenizer, data_dir = data_dir, output_dir = output_dir, divide_chunk = divide_chunk)
    pids = list(range(start * divide_chunk, end * divide_chunk))
    with Pool(len(pids)) as pool:
        pool.map(func, pids)

def read_slimpajama_jsonl(data_path):   
    text, meta = [], []
    with data_path.open('r', encoding='utf-8') as r_f:
        for line in r_f:
            json_data = json.loads(line.strip())
            text.append(json_data["text"])
            meta.append(json_data["meta"])
    return text, meta


def _check_break_point(chunk_dir):
    chunk_dir = Path(chunk_dir)
    print(chunk_dir.stem)
    plist =  list((Path('/rampart-stor/liqun/SlimPajama_jsonl/train/') / chunk_dir.stem).iterdir())
    plist = sorted(plist, key=lambda x: int(x.stem.split("_")[-1]))
    for i in range(0, len(plist)//2):
        if not (chunk_dir / f'{i}.jsonl').exists():
            print(f'{i}.jsonl')
            break
    for i in range(len(plist)//2, len(plist)):
        if not (chunk_dir / f'{i}.jsonl').exists():
            print(f'{i}.jsonl')
            break


        
def _filter_domain(chunk_id, data_root, output_root, suffix):
    data_dir = data_root / f'chunk{chunk_id}'
    output_dir = output_root / f'chunk{chunk_id}'
    output_dir.mkdir(exist_ok=True, parents=True)
    data_path_list = list(data_dir.iterdir())
    data_path_list = sorted(data_path_list, key=lambda x: int(x.stem.split("_")[-1]))
    for data_path in tqdm(data_path_list):
        data = []
        with data_path.open('r', encoding='utf-8') as r_f:
            for line in r_f:
                line_data = line.strip()
                if line_data.endswith(suffix):
                    data.append(line_data)
       
        with (output_dir / data_path.name).open('w', encoding='utf-8') as w_f:
            w_f.write('\n'.join(data))

def filter_domain(data_root, output_root, domain, chunk_num):
    suffix_map = {
        "RedPajamaCommonCrawl": 'l"}}',
        "RedPajamaC4": '4"}}',
        "RedPajamaGithub": 'b"}}',
        "RedPajamaBook": 'k"}}',
        "RedPajamaArXiv": 'v"}}',
        "RedPajamaWikipedia": 'a"}}',
        "RedPajamaStackExchange": 'e"}}',
    }
    data_root = Path(data_root)
    output_root = Path(output_root) / domain
    output_root.mkdir(exist_ok=True, parents=True)
    func = partial(_filter_domain, data_root = data_root, output_root = output_root, suffix = suffix_map[domain])
    chunk_ids = list(range(1, chunk_num+1))
    with Pool(chunk_num) as pool:
        pool.map(func, chunk_ids)

if __name__ == '__main__':
    data_dir = Path('/rampart-stor/liqun/SlimPajama_jsonl/train/')
    output_dir = Path('/rampart-stor/liqun/SlimPajama_tokenized/train/')
    output_dir.mkdir(exist_ok=True, parents=True)
    # tokenize(1, 11, data_dir, output_dir)
    tokenize_omission(data_dir, output_dir)

    # for chunk_dir in output_dir.iterdir():
    #     _check_break_point(chunk_dir)

    # data_root = Path('/rampart-stor/liqun/SlimPajama_jsonl/train/')
    # output_root = Path('/rampart-stor/liqun/SlimPajama_grouped/train/')
    # filter_domain(data_root, output_root, "RedPajamaC4", 10)