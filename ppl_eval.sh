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


CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --model_path record/base/models &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --model_path record/ngram_1_06/models &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --model_path record/ngram_2_06/models &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --model_path record/ngram_4_06/models &
wait

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --model_path record/ngram_8_06/models &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --model_path record/ngram_16_06/models &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --model_path record/ngram_32_06/models &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --model_path record/ngram_16_08/models &
wait


CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --model_path record/ngram_4_06_rs50k/models &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --model_path record/ngram_8_06_rs50k/models &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --model_path record/ngram_16_06_rs50k/models &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --model_path record/ngram_16_08_rs50k/models &
wait


CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --model_path record/normal_06/models &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --model_path record/normal_08/models &
wait