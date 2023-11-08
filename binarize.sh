
# Binarize using fairseq
rm -rf data-bin

TEXT=data
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --joined-dictionary \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20