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

# # base
# python train.py --tag base --smooth_pattern no 

# # n-gram, max_n 32, alpha 0.6
# python train.py --tag ngram_32_06 --smooth_pattern n_gram --max_n 32 --alpha 0.6

# # n-gram, max_n 16, alpha 0.6
# python train.py --tag ngram_16_06 --smooth_pattern n_gram --max_n 16 --alpha 0.6

# n-gram, max_n 8, alpha 0.6
# python train.py --tag ngram_8_06 --smooth_pattern n_gram --max_n 8 --alpha 0.6

# n-gram, max_n 4, alpha 0.6
# python train.py --tag ngram_4_06 --smooth_pattern n_gram --max_n 4 --alpha 0.6

# # n-gram, max_n 2, alpha 0.6
# python train.py --tag ngram_2_06 --smooth_pattern n_gram --max_n 2 --alpha 0.6

# # n-gram, max_n 1, alpha 0.6
# python train.py --tag ngram_1_06 --smooth_pattern n_gram --max_n 1 --alpha 0.6

# # normal, alpha 0.6
# python train.py --tag normal_06 --smooth_pattern normal --alpha 0.6

# # normal, alpha 0.8
# python train.py --tag normal_08 --smooth_pattern normal --alpha 0.8

# n-gram, max_n 16, alpha 0.8
# python train.py --tag ngram_16_08 --smooth_pattern n_gram --max_n 16 --alpha 0.8

# # n-gram, max_n 16, alpha 0.6, RedPajama-Sample-50k
# python train.py --tag ngram_16_06_rs50k --smooth_pattern n_gram --max_n 16 --alpha 0.6 --ngram_dist_name RedPajama-Sample-50k

# # n-gram, max_n 8, alpha 0.6, RedPajama-Sample-50k
# python train.py --tag ngram_8_06_rs50k --smooth_pattern n_gram --max_n 8 --alpha 0.6 --ngram_dist_name RedPajama-Sample-50k

# # n-gram, max_n 4, alpha 0.6, RedPajama-Sample-50k
# python train.py --tag ngram_4_06_rs50k --smooth_pattern n_gram --max_n 4 --alpha 0.6 --ngram_dist_name RedPajama-Sample-50k

# n-gram, max_n 16, alpha 0.8, RedPajama-Sample-50k
# python train.py --tag ngram_16_08_rs50k --smooth_pattern n_gram --max_n 16 --alpha 0.8 --ngram_dist_name RedPajama-Sample-50k


# n-gram, max_n 16, alpha 0.6, alpaca_RedPajama-Sample-50k
python train.py --tag ngram_16_06_alpaca_rs50k --smooth_pattern n_gram --max_n 16 --alpha 0.6 --ngram_dist_name alpaca_RedPajama-Sample-50k