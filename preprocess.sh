
# Binarize using fairseq

fairseq-preprocess \
    --source-lang en \
    --target-lang de \
    --trainpref "train.bpe" \
    --validpref "valid.bpe" \
    --testpref "test.bpe" \
    --destdir "data-bin"