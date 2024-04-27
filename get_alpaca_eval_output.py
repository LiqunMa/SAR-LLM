from tqdm import tqdm
from pathlib import Path
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import transformers
import datasets
import torch
import argparse
from transformers import AutoTokenizer

def alpaca_pipeline(model_name):
    print(f"-- {model_name} --")

    model_p = Path(f"record/{model_name}/models")
    if not model_p.exists():
        model_p = model_name
        

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_p,
        tokenizer = tokenizer,
        torch_dtype=torch.bfloat16,
        use_fast = False,
        device_map='auto',
    )

    if not Path(f"record/{model_name}/models").exists():
        pipeline.model.resize_token_embeddings(len(tokenizer))

        input_embeddings = pipeline.model.get_input_embeddings().weight.data
        output_embeddings = pipeline.model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-1].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-1].mean(dim=0, keepdim=True)

        input_embeddings[-1:] = input_embeddings_avg
        output_embeddings[-1:] = output_embeddings_avg
    return pipeline
    
def get_response(pipe, tokenizer, input_txt):
    sequences = pipe(
        input_txt,
        do_sample=True,
        top_p=1.0,
        temperature=0.7,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2048,
        batch_size=len(input_txt),
        return_full_text=False,
        truncation=True
    )
    res = [item[0]['generated_text'].split('</s>')[0].split('##. Instruction')[0].split('## Instruction')[0] for item in sequences]
    return res


if __name__ == '__main__':
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:{instruction}\n\n### Response:"

    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_name",
        type=str
    )
    args = parser.parse_args()

    special_tokens_dict = dict()
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
    special_tokens_dict["pad_token"] = "[PAD]"
    tokenizer.add_special_tokens(special_tokens_dict)
    
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    dataloader = DataLoader(eval_set, shuffle=False, batch_size=20)

    pipe = alpaca_pipeline(args.model_name)

    save_data = []
    save_dir = Path(f'alpaca_eval_record/{args.model_name.split('/')[-1]}')
    save_dir.mkdir(exist_ok=True, parents=True)
    for batch in tqdm(dataloader):
        inputs = [prompt.format(instruction=ins) for ins in batch['instruction']]
        res = get_response(pipe, tokenizer, inputs)
        for i in range(len(res)):
            item = {
                "instruction": batch['instruction'][i],
                "output": res[i],
                "generator": args.model_name,
                "dataset": batch['dataset'][i]
            }
            save_data.append(item)

        with (save_dir/f"{args.model_name.split('/')[-1]}.json").open('w', encoding='utf-8') as w_f:
            json.dump(save_data, w_f, ensure_ascii=False, indent=4)