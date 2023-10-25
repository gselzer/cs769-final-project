

def main():
    print()

    # punctuation normalization, tokenization, data cleaning, and true-casing via Moses

    # sample commands from manual (english and french)

    # tokenization
    # ~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < ~/corpus/training/news-commentary-v8.fr-en.en > ~/corpus/news-commentary-v8.fr-en.tok.en
    # ~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l fr < ~/corpus/training/news-commentary-v8.fr-en.fr > ~/corpus/news-commentary-v8.fr-en.tok.fr

    # true-casing
    # ~/mosesdecoder/scripts/recaser/train-truecaser.perl --model ~/corpus/truecase-model.en --corpus ~/corpus/news-commentary-v8.fr-en.tok.en
    # ~/mosesdecoder/scripts/recaser/train-truecaser.perl --model ~/corpus/truecase-model.fr --corpus ~/corpus/news-commentary-v8.fr-en.tok.fr

    # ~/mosesdecoder/scripts/recaser/truecase.perl --model ~/corpus/truecase-model.en < ~/corpus/news-commentary-v8.fr-en.tok.en > ~/corpus/news-commentary-v8.fr-en.true.en
    # ~/mosesdecoder/scripts/recaser/truecase.perl --model ~/corpus/truecase-model.fr < ~/corpus/news-commentary-v8.fr-en.tok.fr > ~/corpus/news-commentary-v8.fr-en.true.fr

    # clean, limiting sentence length (this example, to 80)
    # ~/mosesdecoder/scripts/training/clean-corpus-n.perl ~/corpus/news-commentary-v8.fr-en.true fr en ~/corpus/news-commentary-v8.fr-en.clean 1 80

import re
import random

# This pattern will search for any tags
tag_pattern = re.compile(r'<.*?>')

# Read in the files
with open("de-en/train.tags.de-en.de") as f:
    text_de = f.read().split('\n')
with open("de-en/train.tags.de-en.en") as f:
    text_en = f.read().split('\n')

# Filter out any strings that have a tag
xmlless_de = [s for s in text_de if not re.search(tag_pattern, s)]
xmlless_en = [s for s in text_en if not re.search(tag_pattern, s)]

# Randomly sample from the xml-less lines in the files
selected = random.sample([x for x in zip(xmlless_de, xmlless_en)], 5)

selected_de, selected_en = list(zip(*selected))

with open("train.sample.de", "w") as f:
    f.write("\n".join(selected_de))
    f.write("\n")
with open("train.sample.en", "w") as f:
    f.write("\n".join(selected_en))
    f.write("\n")
