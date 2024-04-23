lm_eval --model vllm \
    --model_args pretrained=huggyllama/llama-7b,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks gsm8k,truthfulqa_mc2\
    --batch_size auto

lm_eval --model vllm \
    --model_args pretrained=record/base/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks gsm8k,truthfulqa_mc2\
    --batch_size auto

lm_eval --model vllm \
    --model_args pretrained=record/ngram_32_06/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks gsm8k,truthfulqa_mc2\
    --batch_size auto

lm_eval --model vllm \
    --model_args pretrained=record/ngram_16_06/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks gsm8k,truthfulqa_mc2\
    --batch_size auto

lm_eval --model vllm \
    --model_args pretrained=record/ngram_8_06/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks gsm8k,truthfulqa_mc2 \
    --batch_size auto


# lm_eval --model vllm \
#     --model_args pretrained=record/normal_06/models,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
#     --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa \
#     --batch_size auto