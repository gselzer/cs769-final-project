import random
import os
from typing import List

def mRASP(
        src_files: List[str],
        tgt_files: List[str],
        output_dir: str,
        replacement_prob: float = 0.3,
        src_lang: str = "de",
        tgt_lang: str = "en",
    ):
    # Download dictionary
    dictionary = download_dictionary(src_lang, tgt_lang)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for src_file in src_files:
        with open(src_file, "r") as f:
            src = f.readlines()
        
        new_src = []
        for line in src:
            new_tokens = []
            for token in line.split():
                if token in dictionary and random.random() < replacement_prob:
                    token = random.choice(dictionary[token])
                new_tokens.append(token)
            new_src.append(" ".join(new_tokens) + "\n")

        src_file = os.path.basename(src_file)
        if src_file.startswith("tmp."):
            src_file = src_file.replace("tmp.", "")


        with open(os.path.join(output_dir, f"{src_file}"), "w") as f:
            f.writelines(new_src)

    for tgt_file in tgt_files:
        t = os.path.basename(tgt_file)
        if t.startswith("tmp."):
            t = t.replace("tmp.", "")
        foo = os.system(f"cp {tgt_file} {os.path.join(output_dir, t)}")
        print(foo)

def download_dictionary(src_lang: str, tgt_lang: str):
    filename = "./dictionary.txt"
    if not os.path.exists(filename):
        dictionary_url = f"https://dl.fbaipublicfiles.com/arrival/dictionaries/{tgt_lang}-{src_lang}.txt"
        os.system(f"wget {dictionary_url} -O {filename}")
    with open(filename, "r") as f:
        lines = [l.split() for l in f.readlines()]
    src = [l[1] for l in lines]
    tgt = [l[0] for l in lines]

    dictionary = {}
    for s, t in zip(src, tgt):
        dictionary.setdefault(s, []).append(t)
    
    return dictionary


if __name__ == "__main__":
    mRASP("de-en/train.tags.de-en.de", "de-en/train.tags.de-en.en")
    