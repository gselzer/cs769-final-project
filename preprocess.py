import argparse
import os
import shutil
from techniques.interleaving.linguistic import tag
from techniques.mRASP.mrasp import mRASP
from techniques.mBART.mbart import mBART
import utils

# Configuration Constants

def preprocess(args: argparse.Namespace):
    DATA_DIR = "data/"
    TEMP_DIR = "temp/"
    DELETE_TEMP_DATA = True
    SRC_LANG = args.src
    TGT_LANG = args.tgt
    PRETRAIN = False

    # Delete data from old runs in "data" and "temp" directories
    for dir_path in [TEMP_DIR, DATA_DIR]:
        if os.path.exists(dir_path):
            os.system(f"rm -rf {dir_path}")

    # Make temp and data directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('temp', exist_ok=True)

    utils.download_data(TEMP_DIR, SRC_LANG, TGT_LANG)
    utils.preprocess_data(TEMP_DIR, SRC_LANG, TGT_LANG)

    # mRASP
    if args.mRASP:
        PRETRAIN = True
        mRASP(
            [f"{TEMP_DIR}tmp.train.{SRC_LANG}", f"{TEMP_DIR}tmp.valid.{SRC_LANG}", f"{TEMP_DIR}tmp.test.{SRC_LANG}"],
            [f"{TEMP_DIR}tmp.train.{TGT_LANG}", f"{TEMP_DIR}tmp.valid.{TGT_LANG}", f"{TEMP_DIR}tmp.test.{TGT_LANG}"],
            f"{TEMP_DIR}pretrain"
        )

    # POS tag data
    if args.part_of_speech:
        for s in ["train", "test", "valid"]:
            # Only add POS for source data
            for l in [SRC_LANG]:
                tag(f"{TEMP_DIR}tmp.{s}.{l}", l)

    # mBART
    if args.mBART:
        PRETRAIN = True
        mBART( TEMP_DIR, f"{TEMP_DIR}pretrain", SRC_LANG, TGT_LANG)

    # Learn BPE
    num_bpe_tokens: int = 10000
    os.system(f"subword-nmt learn-joint-bpe-and-vocab --input {TEMP_DIR}tmp.train.{TGT_LANG} {TEMP_DIR}tmp.train.{SRC_LANG} \
                -s {num_bpe_tokens} -o {TEMP_DIR}code.txt --write-vocabulary {TEMP_DIR}vocab.{TGT_LANG} {TEMP_DIR}vocab.{SRC_LANG}")
    
    if args.joint_dropout:
        if args.gen_jd_data:
            # Joint Dropout
            wa = utils.WordAligner()
            wa.word_alignments(source_file=f"{TEMP_DIR}tmp.train.{SRC_LANG}",
                                target_file=f"{TEMP_DIR}tmp.train.{TGT_LANG}",
                                output_file=f'{TEMP_DIR}eflomal.{SRC_LANG}.{TGT_LANG}',
                                model = '3')
            JDR = utils.JointDropout(debug=False)
            JDR.joint_dropout(f"{TEMP_DIR}tmp.train.{SRC_LANG}", f"{TEMP_DIR}tmp.train.{TGT_LANG}", 
                              f'{TEMP_DIR}eflomal.{SRC_LANG}.{TGT_LANG}', output_dir=f'{TEMP_DIR}', 
                              src_suffix=SRC_LANG,trg_suffix=TGT_LANG)

            # Concatenate JDR output with the tmp.train.TGT_LANG/tmp.train.SRC_LANG files.
            with open(f'{TEMP_DIR}jdr.src.{SRC_LANG}', 'r') as source_file:
                data_to_append = source_file.read()

            with open(f'{TEMP_DIR}tmp.train.{SRC_LANG}', 'a') as target_file:
                target_file.write(data_to_append)

            with open(f'{TEMP_DIR}jdr.trg.{TGT_LANG}', 'r') as source_file:
                data_to_append = source_file.read()

            with open(f'{TEMP_DIR}tmp.train.{TGT_LANG}', 'a') as target_file:
                target_file.write(data_to_append)
        else:
            if SRC_LANG == 'ne':
                shutil.copy('data-jd/tmp.train.ne-en.20k.en','temp/tmp.train.en')
                shutil.copy('data-jd/tmp.train.ne-en.20k.ne','temp/tmp.train.ne')
            elif SRC_LANG == 'de':
                shutil.copy('data-jd/tmp.train.de-en.10k.en','temp/tmp.train.en')
                shutil.copy('data-jd/tmp.train.de-en.10k.de','temp/tmp.train.de')
            else:
                raise Exception('source language needs to be one of "ne" or "de"')
            

    # Apply BPE
    glossary = [
        # Joint Dropout
        "<S_\d+>",
        "<T_\d+>",
        # Penn Treebank POS tags
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "<mask>"
    ]
    glossary_str = " ".join([f"'{d}'" for d in glossary])
    for s in ["train", "test", "valid"]:
        for l in [SRC_LANG, TGT_LANG]:
            os.system(f"subword-nmt apply-bpe -c {TEMP_DIR}code.txt --vocabulary {TEMP_DIR}vocab.{l} < \
                        {TEMP_DIR}tmp.{s}.{l} > {DATA_DIR}{s}.{l} --glossaries {glossary_str}")

    # Apply BPE to pretrained data
    if PRETRAIN:
        os.makedirs(f"{DATA_DIR}pretrain")
        for s in ["train", "test", "valid"]:
            for l in [SRC_LANG, TGT_LANG]:
                os.system(f"subword-nmt apply-bpe -c {TEMP_DIR}code.txt --vocabulary {TEMP_DIR}vocab.en < \
                            {TEMP_DIR}pretrain/{s}.{l} > {DATA_DIR}pretrain/{s}.{l} --glossaries {glossary_str}")

    os.system(f"cp {TEMP_DIR}code.txt {DATA_DIR}code.txt")

    # Run fairseq-preprocess
    if os.path.exists("data-bin"):
        shutil.rmtree("data-bin")
    preprocess_command = f""" fairseq-preprocess --source-lang {SRC_LANG} --target-lang {TGT_LANG} --trainpref {DATA_DIR}train --validpref {DATA_DIR}valid --testpref {DATA_DIR}test --destdir data-bin --joined-dictionary --workers 20"""
    os.system(preprocess_command)
    os.system(f"cp data-bin/dict.{SRC_LANG}.txt data/join-dict.txt")

    # Delete Temp Files
    if DELETE_TEMP_DATA:
        os.system("rm -rf temp")
    # Delete Any Existing Checkpoints
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess training data.")
    # Parse source and target languages
    parser.add_argument('--src', choices=["de", "ne"], help='the source language')
    parser.add_argument('--tgt', choices=["en"], help='the target language')
    # Parse techniques
    for technique in ['joint-dropout', 'mRASP', 'part-of-speech', 'mBART', 'gen-jd-data']:
        parser.add_argument(f'--{technique}', action=argparse.BooleanOptionalAction)

    preprocess(parser.parse_args())
