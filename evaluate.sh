fairseq-generate data-bin \
    --path checkpoints/checkpoint_best.pt \
    --max-tokens 4096 --beam 5 \
    --results-path results \
    --sacrebleu