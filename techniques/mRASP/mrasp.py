import random
import os
import string
from typing import Dict, List

def _mRASP(
        src_files: List[str],
        tgt_files: List[str],
        dictionary: Dict[str, List[str]],
        output_dir: str,
        src_lang,
        tgt_lang,
        replacement_prob: float = 0.3,
    ):
    # Download dictionary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stats = {}
    for src_file in src_files:
        with open(src_file, "r") as f:
            src = f.read().split("\n")
        
        stats[src_file] = dict(
            total = 0,
            translated = 0
        )
        
        new_src = []
        for line in src:
            tokens = line.split()
            if tokens == []:
                continue
            # NB Dictionary quality is generally poor for low-resource languages.
            # Many words may not be represented in the dictionary.
            # Here we try our best to adjust the replacement probability
            words = [(i, x) for i, x in enumerate(tokens) if any(c not in string.punctuation for c in x)]

            translatable = [(i, word) for i, word in words if word in dictionary]
            k = round(min(len(translatable), replacement_prob * len(words)))
            sampled = random.sample(translatable, round(k))
            for idx, word in sampled:
                tokens[idx] = random.choice(dictionary[word])
            
            stats[src_file]["total"] += len(words)
            stats[src_file]["translated"] += k
            new_src.append(" ".join(tokens))
            new_src.append(line)
            
        src_file = os.path.basename(src_file)
        if src_file.startswith("tmp."):
            src_file = src_file.replace("tmp.", "")
        with open(os.path.join(output_dir, f"{src_file}"), "w") as f:
            f.write("\n".join(new_src))
    
    for tgt_file in tgt_files:
        with open(tgt_file) as f:
            tgt = f.read().split("\n")
        t = os.path.basename(tgt_file)
        if t.startswith("tmp."):
            t = t.replace("tmp.", "")
        new_tgt = []
        for line in tgt:
            if line.split() == []:
                continue
            new_tgt.append(line)
            new_tgt.append(line)
        with open(os.path.join(output_dir, t), "w") as f:
            f.write("\n".join(new_tgt))
    
    with open(f"mrasp_stats.{src_lang}-{tgt_lang}", "w") as f:
        f.write(f"{src_lang}-{tgt_lang} RASP using dictionary with {len(dictionary)} words")
        for filename in stats.keys():
            f.write(f"{filename} - \n")
            n = stats[filename]['translated']
            d = stats[filename]['total']
            f.write(f"\tnumber of tokens seen: {d}\n")
            f.write(f"\tnumber of tokens translated: {n}\n")
            f.write(f"\tProportion: {n/d}\n")


def mRASP(
        src_files: List[str],
        tgt_files: List[str],
        output_dir: str,
        replacement_prob: float = 0.3,
        src_lang: str = "de",
        tgt_lang: str = "en",
    ):
    if src_lang == "de" and tgt_lang == "en":
        dictionary = download_de_en_dictionary()
    elif src_lang == "ne" and tgt_lang == "en":
        dictionary = download_ne_en_dictionary()
    else:
        raise Exception(f"mRASP not implemented in {src_lang}->{tgt_lang} translation!")

    _mRASP(src_files, tgt_files, dictionary, output_dir, src_lang, tgt_lang, replacement_prob)

def download_de_en_dictionary():
    filename = "de_en/dictionary.txt"
    if not os.path.exists(filename):
        if not os.path.exists("de_en"):
            os.makedirs("de_en")
        dictionary_url = f"https://dl.fbaipublicfiles.com/arrival/dictionaries/de-en.txt"
        os.system(f"wget {dictionary_url} -O {filename}")
    with open(filename, "r") as f:
        lines = [l.split() for l in f.readlines()]
    src = [l[0] for l in lines]
    tgt = [l[1] for l in lines]

    dictionary = {}
    for s, t in zip(src, tgt):
        dictionary.setdefault(s, []).append(t)
    
    return dictionary

def download_ne_en_dictionary():
    dictionary = {}
    import csv
    filename = "dictionary.txt"
    if not os.path.exists(filename):
        dictionary_url = "https://raw.githubusercontent.com/nirooj56/Nepdict/master/database/data.csv"
        os.system(f"wget {dictionary_url} -O {filename}")
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for line in reader:
            word_en = line[0]
            # Idx 1 is a comma, 2 is POS, 3 is comma, 4+ are Nepali translations
            cleaned = [s.replace('"', '').replace(',', "") for s in line[4:]]
            for word_ne in cleaned:
                dictionary.setdefault(word_ne, []).append(word_en)
    
    return dictionary

if __name__ == "__main__":
    TEMP_DIR = ""
    SRC_LANG = "ne"
    TGT_LANG = "en"
    mRASP(
        [f"{TEMP_DIR}tmp.train.{SRC_LANG}", f"{TEMP_DIR}tmp.valid.{SRC_LANG}", f"{TEMP_DIR}tmp.test.{SRC_LANG}"],
        [f"{TEMP_DIR}tmp.train.{TGT_LANG}", f"{TEMP_DIR}tmp.valid.{TGT_LANG}", f"{TEMP_DIR}tmp.test.{TGT_LANG}"],
        f"{TEMP_DIR}pretrain",
        src_lang=SRC_LANG,
    )
    