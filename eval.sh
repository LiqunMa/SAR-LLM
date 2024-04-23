lm_eval --model hf \
    --model_args pretrained=record/base/models,parallelize=True \
    --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa \
    --batch_size 64