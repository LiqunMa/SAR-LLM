# lm_eval --model vllm \
#     --model_args pretrained=huggyllama/llama-7b,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks gsm8k,truthfulqa_mc2\
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/base/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks gsm8k,truthfulqa_mc2\
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/ngram_32_06/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks gsm8k,truthfulqa_mc2\
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/ngram_16_06/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks gsm8k,truthfulqa_mc2\
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/ngram_8_06/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks gsm8k,truthfulqa_mc2 \
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/ngram_4_06/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks gsm8k,truthfulqa_mc2 \
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/ngram_2_06/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks gsm8k,truthfulqa_mc2 \
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/normal_06/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks gsm8k,truthfulqa_mc2 \
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/normal_08/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks gsm8k,truthfulqa_mc2 \
#     --batch_size auto


# lm_eval --model vllm \
#     --model_args pretrained=record/ngram_1_06/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,gsm8k,truthfulqa_mc2 \
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/ngram_16_08/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,gsm8k,truthfulqa_mc2 \
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/ngram_4_06_rs50k/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,gsm8k,truthfulqa_mc2 \
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/ngram_8_06_rs50k/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,gsm8k,truthfulqa_mc2 \
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/ngram_16_06_rs50k/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,gsm8k,truthfulqa_mc2 \
#     --batch_size auto

# lm_eval --model vllm \
#     --model_args pretrained=record/ngram_16_08_rs50k/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,gsm8k,truthfulqa_mc2 \
#     --batch_size auto

lm_eval --model vllm \
    --model_args pretrained=record/ngram_16_06_alpaca_rs50k/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,gsm8k,truthfulqa_mc2 \
    --batch_size auto