import json
from pathlib import Path
from tqdm import tqdm

def get_max_len(data_path):
    max_len = 0
    with Path(data_path).open('r', encoding='utf-8') as r_f:
        
        for line in tqdm(r_f):
            d = json.loads(line.strip())
            max_len = max(max_len, len(d['tk_ids']))
    print(max_len)

def stat_file_ngram(file_p, n_gram=1, key='tk_ids', add_eos_id=False):
    n_gram_dict = {}
    prefix_dict = {}
    with Path(file_p).open('r', encoding='utf-8') as r_f:
        for line in tqdm(r_f):
            d = json.loads(line.strip())
            if add_eos_id:
                tk_ids = d[key] + [2]
            else:
                tk_ids = d[key]
            for i in range(len(tk_ids) - n_gram + 1):
                n_gram_tk = tk_ids[i: i + n_gram]
                prefix = ' '.join([str(t) for t in n_gram_tk[:-1]])
                if prefix not in prefix_dict:
                    prefix_dict[prefix] = set()
                
                n_gram_tk = ' '.join( [str(t) for t in n_gram_tk] )
                prefix_dict[prefix].add(n_gram_tk.split()[-1])
                
                if n_gram_tk in n_gram_dict:
                    n_gram_dict[n_gram_tk] += 1
                else:
                    n_gram_dict[n_gram_tk] = 1
    ngram_num = sum(n_gram_dict.values())
    res = {
        'n_gram_num': ngram_num,
        'n_gram_vsize': len(n_gram_dict),
        'n_gram_dict': {}
    }
    for k, v in n_gram_dict.items():
        _prefix = ' '.join(k.split()[:-1])
        if len(prefix_dict[_prefix]) > 1:
            res['n_gram_dict'][k] = v 
    res['n_gram_num_p>1']   = sum(res['n_gram_dict'].values())
    res['n_gram_vsize_p>1'] = len(res['n_gram_dict'])
    return res

def get_all_ngram_count(tk_file_path = 'tokenized_data/alpaca_tokenized.json', save_name='alpaca', max_n=32, key='tk_ids', add_eos_id=False):
    record = {}
    Path(f'n_gram_dicts/{save_name}').mkdir(exist_ok=True, parents=True)
    for n_gram in tqdm(range(1, max_n+1)):
        res = stat_file_ngram(tk_file_path, n_gram, key = key, add_eos_id=add_eos_id)
        if res["n_gram_num_p>1"] == 0:
            break
        print(f'n_gram: {n_gram}, n_gram_vsize: {res["n_gram_vsize"]}, n_gram_num: {res["n_gram_num"]}, n_gram_vsize_p>1: {res["n_gram_vsize_p>1"]}, n_gram_num_p>1: {res["n_gram_num_p>1"]}')
        with Path(f'n_gram_dicts/{save_name}/{n_gram}_gram.json').open('w', encoding='utf-8') as w_f:
            json.dump(res, w_f, ensure_ascii=False)
        record[f'{n_gram}_gram'] = {
            'n_gram_vsize': res["n_gram_vsize"],
            'n_gram_num': res["n_gram_num"],
            'n_gram_vsize_p>1': res["n_gram_vsize_p>1"],
            'n_gram_num_p>1': res["n_gram_num_p>1"]
        }
        with Path(f'n_gram_dicts/record/{save_name}.json').open('w', encoding='utf-8') as w_f:
            json.dump(record, w_f, ensure_ascii=False, indent=4)

# def get_prefix_set(pre_ngram, ordered_ngram_list):
#     prefix = pre_ngram + ' '
#     ngram_with_prefix_set = set()
#     state = False
#     for ngram in ordered_ngram_list:
#         if ngram.startswith(prefix):
#             state = True
#             ngram_with_prefix_set.add(ngram)
#         else:
#             if state == True:
#                 break
#     return ngram_with_prefix_set

def get_infi_ngram_distribution(data_root, max_n_gram=382):
    data_root = Path(data_root)
    save_dir = Path('n_gram_distribution')
    save_dir.mkdir(exist_ok=True)

    infi_ngram_distribution = {
        n: {} 
        for n in range(2, max_n_gram + 1)
    }
    with (data_root/f'1_gram.json').open('r', encoding='utf-8') as r_f:
        ngram_count = json.load(r_f)['n_gram_dict']
    
    _tks = [k for k in ngram_count]
    _probs = [ngram_count[k] for k in ngram_count]
    total = sum(_probs)
    _probs = [p / total for p in _probs]
    infi_ngram_distribution['1'] = [_tks, _probs]

    for n in tqdm(range(2, max_n_gram + 1)):
        with (data_root/f'{n}_gram.json').open('r', encoding='utf-8') as r_f:
            ngram_count = json.load(r_f)['n_gram_dict']
        sorted_ngrams = sorted(ngram_count.keys())

        dist_dict = {}
        global_prefix = ' '.join(sorted_ngrams[0].split()[:-1])
        _next_tks, _prob = [], []
        for ngram in tqdm(sorted_ngrams):
            _prefix = ' '.join(ngram.split()[:-1])
            if _prefix != global_prefix:
                if len(_next_tks) > 1:
                    _next_tks = [int(i) for i in _next_tks]
                    total_tk_num = sum(_prob)
                    _prob = [p / total_tk_num for p in _prob]
                    dist_dict[global_prefix] = [
                        _next_tks,
                        _prob
                    ]
                _next_tks, _prob = [], []
                global_prefix = _prefix
            
            _next_tks.append(ngram.split()[-1])
            _prob.append(ngram_count[ngram])
        
        assert len(_next_tks) != 0
        if len(_next_tks) > 1:
            _next_tks = [int(i) for i in _next_tks]
            total_tk_num = sum(_prob)
            _prob = [p / total_tk_num for p in _prob]
            
            dist_dict[global_prefix] = [
                        _next_tks,
                        _prob
                    ]

        infi_ngram_distribution[str(n)] = dist_dict

    with (save_dir/f'{data_root.stem}_distribution.json').open('w', encoding='utf-8') as w_f:
        json.dump(infi_ngram_distribution, w_f, ensure_ascii=False)
    

def check():
    with Path('n_gram_distribution/alpaca_distribution.json').open('r', encoding='utf-8') as r_f:
        dist = json.load(r_f)
    for n in range(32, 33):
        with Path(f'n_gram_dicts/alpaca/{n}_gram.json').open('r', encoding='utf-8') as r_f:
            ngram_count = json.load(r_f)['n_gram_dict']
        with Path('a.json').open('w') as w_f:
            json.dump(dist[str(n)], w_f, ensure_ascii=False, indent=4)
        with Path('b.json').open('w') as w_f:
            json.dump(ngram_count, w_f, ensure_ascii=False, indent=4)
        # print('dist', dist[str(n)])
        # print('ngram_count', ngram_count)

if __name__ == '__main__':
    # get_max_len('tokenized_data/alpaca_tokenized.json')
    # get_all_ngram_count()
    # get_infi_ngram_distribution('n_gram_dicts/alpaca', max_n_gram=32)
    
    get_all_ngram_count(tk_file_path = 'tokenized_data/RedPajama-Data-1T-Sample_all.jsonl', save_name='RedPajama-Sample', max_n=32, key="input_ids", add_eos_id=True)
    get_infi_ngram_distribution('n_gram_dicts/RedPajama-Sample', max_n_gram=32)