

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
