#!/bin/bash
#SBATCH --reservation=experiment
#SBATCH --job-name=test
#SBATCH --partition=gpumid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err


# CUDA_VISIBLE_DEVICES=0 python get_alpaca_eval_output.py --model_name base &
# CUDA_VISIBLE_DEVICES=1 python get_alpaca_eval_output.py --model_name ngram_2_06 &
# CUDA_VISIBLE_DEVICES=2 python get_alpaca_eval_output.py --model_name ngram_4_06 &
# CUDA_VISIBLE_DEVICES=3 python get_alpaca_eval_output.py --model_name ngram_8_06 &
# wait

# CUDA_VISIBLE_DEVICES=0 python get_alpaca_eval_output.py --model_name ngram_16_06 &
# CUDA_VISIBLE_DEVICES=1 python get_alpaca_eval_output.py --model_name ngram_32_06 &
# CUDA_VISIBLE_DEVICES=2 python get_alpaca_eval_output.py --model_name ngram_16_08 &
# CUDA_VISIBLE_DEVICES=3 python get_alpaca_eval_output.py --model_name normal_06 &
# wait

# CUDA_VISIBLE_DEVICES=0 python get_alpaca_eval_output.py --model_name normal_08 &
# CUDA_VISIBLE_DEVICES=1 python get_alpaca_eval_output.py --model_name ngram_1_06 &
# CUDA_VISIBLE_DEVICES=2 python get_alpaca_eval_output.py --model_name ngram_4_06_rs50k &
# CUDA_VISIBLE_DEVICES=3 python get_alpaca_eval_output.py --model_name ngram_8_06_rs50k &
# wait

# CUDA_VISIBLE_DEVICES=0 python get_alpaca_eval_output.py --model_name ngram_16_06_rs50k &
# CUDA_VISIBLE_DEVICES=1 python get_alpaca_eval_output.py --model_name ngram_16_08_rs50k &
# wait

python get_alpaca_eval_output.py --model_name huggyllama/llama-7b