import os
import random
from typing import List
from mosestokenizer import MosesPunctuationNormalizer, MosesTokenizer
from subword_nmt import learn_bpe, apply_bpe
from sample import read_source_and_target


# Sample source and target
all_en, all_de = read_source_and_target("de-en/train.tags.de-en.en", "de-en/train.tags.de-en.de")

# Normalize sampled source and target
norm_en = MosesPunctuationNormalizer(lang="en")
norm_de = MosesPunctuationNormalizer(lang="de")
normalized_en = [norm_en(s) for s in all_en]
normalized_de = [norm_de(s) for s in all_de]

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
tmp_file_en = "cleaned.en"
tmp_file_de = "cleaned.de"
with open(tmp_file_en, "w") as f:
    f.write("\n".join([s.lower() for s in normalized_en]))
    f.write("\n")
with open(tmp_file_de, "w") as f:
    f.write("\n".join([s.lower() for s in normalized_de]))
    f.write("\n")

# Learn BPE
num_bpe_tokens: int = 10000
codes_file_en = "codes.en"
codes_file_de = "codes.de"
# TODO: For some reason, I got "Permission denied" when trying to use subprocess - WHY?
os.system(f"subword-nmt learn-bpe -s {num_bpe_tokens} < {tmp_file_en} > {codes_file_en}")
os.system(f"subword-nmt learn-bpe -s {num_bpe_tokens} < {tmp_file_de} > {codes_file_de}")
os.remove(tmp_file_en)
os.remove(tmp_file_de)

# Apply BPE to training and validation datasets
## HyperParameters

def apply_sample_bpe(purpose: str, indices: List[int]):
    sample_en = [all_en[i] for i in indices]
    sample_de = [all_de[i] for i in indices]
    with open("tmp.en", "w") as f:
        f.write("\n".join(sample_en))
        f.write("\n")
    with open("tmp.de", "w") as f:
        f.write("\n".join(sample_de))
        f.write("\n")

    os.system(f"subword-nmt apply-bpe -c {codes_file_en} < tmp.en > {purpose}.bpe.en")
    os.system(f"subword-nmt apply-bpe -c {codes_file_de} < tmp.de > {purpose}.bpe.de")

    os.remove("tmp.en")
    os.remove("tmp.de")
    

n_train: int = 10000
n_validate: int = 1000
n_test: int = 1000

selected_idx =  random.sample(range(len(all_en)), n_train + n_validate + n_test)
apply_sample_bpe("train", selected_idx[:n_train])
apply_sample_bpe("valid", selected_idx[n_train:n_train+n_validate])
apply_sample_bpe("test", selected_idx[n_train+n_validate:])
