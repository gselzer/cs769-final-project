import os
from mosestokenizer import MosesPunctuationNormalizer, MosesTokenizer
from subword_nmt import learn_bpe, apply_bpe
from sample import sample_source_and_target

## HyperParameters
num_samples: int = 100
num_bpe_tokens: int = 100

# Sample source and target
selected_en, selected_de = sample_source_and_target("de-en/train.tags.de-en.en", "de-en/train.tags.de-en.de", num_samples)

# Normalize sampled source and target
norm_en = MosesPunctuationNormalizer(lang="en")
norm_de = MosesPunctuationNormalizer(lang="de")
normalized_en = [norm_en(s) for s in selected_en]
normalized_de = [norm_de(s) for s in selected_de]

# Tokenize sampled source and target
tokenizer_en = MosesTokenizer(lang="en")
tokenizer_de = MosesTokenizer(lang="de")
token_en = [tokenizer_en(s) for s in normalized_en]
token_de = [tokenizer_de(s) for s in normalized_de]

# TODO: Data cleaning?
# Here's what the other paper did: ~/mosesdecoder/scripts/training/clean-corpus-n.perl ~/corpus/news-commentary-v8.fr-en.true fr en ~/corpus/news-commentary-v8.fr-en.clean 1 80

# Truecasing - instead, for now, let's just lowercase stuff
# BUT here is what the other paper did
# ~/mosesdecoder/scripts/recaser/train-truecaser.perl --model ~/corpus/truecase-model.en --corpus ~/corpus/news-commentary-v8.fr-en.tok.en
# ~/mosesdecoder/scripts/recaser/train-truecaser.perl --model ~/corpus/truecase-model.fr --corpus ~/corpus/news-commentary-v8.fr-en.tok.fr
# ~/mosesdecoder/scripts/recaser/truecase.perl --model ~/corpus/truecase-model.en < ~/corpus/news-commentary-v8.fr-en.tok.en > ~/corpus/news-commentary-v8.fr-en.true.en
# ~/mosesdecoder/scripts/recaser/truecase.perl --model ~/corpus/truecase-model.fr < ~/corpus/news-commentary-v8.fr-en.tok.fr > ~/corpus/news-commentary-v8.fr-en.true.fr
truecased_en = [[t.lower() for t in s] for s in token_en]
truecased_de = [[t.lower() for t in s] for s in token_de]

# Apply Joint Dropout

# Write out intermediate data
tmp_file_en = "train.sample.en"
with open(tmp_file_en, "w") as f:
    f.write("\n".join(normalized_en))
    f.write("\n")
tmp_file_de = "train.sample.de"
with open(tmp_file_de, "w") as f:
    f.write("\n".join(normalized_de))
    f.write("\n")

# Apply BPE
codes_file_en = "train.codes.en"
bpe_file_en = "train.bpe.en"
# TODO: For some reason, I got "Permission denied" when trying to use subprocess - WHY?
os.system(f"subword-nmt learn-bpe -s {num_bpe_tokens} < {tmp_file_en} > {codes_file_en}")
os.system(f"subword-nmt apply-bpe -c {codes_file_en} < {tmp_file_en} > {bpe_file_en}")

codes_file_de = "train.codes.de"
bpe_file_de = "train.bpe.de"
os.system(f"subword-nmt learn-bpe -s {num_bpe_tokens} < {tmp_file_de} > {codes_file_de}")
os.system(f"subword-nmt apply-bpe -c {codes_file_de} < {tmp_file_de} > {bpe_file_de}")