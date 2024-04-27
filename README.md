# SAR-LLM
Code: [https://github.com/LiqunMa/SAR-LLM](https://github.com/LiqunMa/SAR-LLM)

## How to finetuning with soft targets

### Requirements
accelerate==0.29.3

datasets==2.19.0

lm_eval==0.4.2

tokenizers==0.19.1

torch==2.2.2

transformers==4.40.0

wandb==0.16.2

vllm==0.3.2

### Download data
1. Alpaca: [https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json). Please save in `finetuning_data/`.
2. RedPajama-Data-1T-Sample: [https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample). Please save in `finetuning_data/Redpajama-Sample`.

### Preprocess data
To get the tokenized data for calculate the n-gram distribution:

`python preprocess_data.py` 

### Calculate the n-gram distribution
`python n_gram.py`

### Finetuning model
Please adjust the specific configuration parameters based on your Slurm setup:

`sbatch sbatch.sh`

### Eval downstream task
`bash eval_vllm.sh`

### Eval PPL
`bash eval_vllm.sh`

### AlpacaEval
Get the output of eval dataset:

`bash alpaca_eval_output.sh`

Get the win rate:

`alpaca_eval.sh`

For more information, please refer to [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)。
