fairseq-generate data-bin \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 \
    --results-path results \
    --sacrebleu