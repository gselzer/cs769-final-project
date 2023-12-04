from numpy.random import poisson
import random
import os
from typing import List

def mBART(
        src_dir: str, 
        output_dir: str, 
        src_lang: str,
        tgt_lang: str,
        lambda_value: float = 3.5):
 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # concatenate source and target langs, place noised into source and not noised into target
    concatenation = []
    for lang in [src_lang, tgt_lang] :
        file = os.path.join(src_dir, f"tmp.train.{lang}")
        with open(file, "r") as f:
            for s in f.read().split("\n"):
                concatenation.append(s)
            # sentences = f.read().split("\n")

    noised_concatenation = [noising(sentence, lambda_value) for sentence in concatenation]


    src_file = os.path.basename(os.path.join(src_dir, f"tmp.train.{src_lang}"))
    if src_file.startswith("tmp."):
        src_file = src_file.replace("tmp.", "")

    tgt_file = os.path.basename(os.path.join(src_dir, f"tmp.train.{tgt_lang}"))
    if tgt_file.startswith("tmp."):
        tgt_file = tgt_file.replace("tmp.", "")

    # noised --> source
    with open(os.path.join(output_dir, f"{src_file}"), "w") as f:
        f.write("\n".join(noised_concatenation))

    # not noised --> target
    with open(os.path.join(output_dir, f"{tgt_file}"), "w") as f:
        f.write("\n".join(concatenation))

    # # Noise training data
    # for lang in [src_lang, tgt_lang] :
    #     file = os.path.join(src_dir, f"tmp.train.{lang}")
    #     with open(file, "r") as f:
    #         sentences = f.read().split("\n")
        
    #     noised_sentences = [noising(sentence, lambda_value) for sentence in sentences]

    #     src_file = os.path.basename(file)
    #     if src_file.startswith("tmp."):
    #         src_file = src_file.replace("tmp.", "")

    #     with open(os.path.join(output_dir, f"{src_file}"), "w") as f:
    #         f.write("\n".join(noised_sentences))
        
    # # Copy test/validation data
    # for lang in [src_lang, tgt_lang]:
    #     for s in ["valid", "test"]:
    #         os.system(f"cp {src_dir}tmp.{s}.{lang} {os.path.join(output_dir, f'{s}.{lang}')}")


    # Copy test/validation concatenations
    for step in ["valid", "test"]:
        tv_concat = []
        for lang in [src_lang, tgt_lang]:
            with open(f"{src_dir}tmp.{step}.{lang}", "r") as f:
                for s in f.read().split("\n"):
                    tv_concat.append(s)
        with open(f"{os.path.join(output_dir, f'{step}.{lang}')}", "w") as f:
            f.write("\n".join(tv_concat))

    

def noising(sentence: str, lambda_value: float):
    """
    Apply mBART-style noising to a sentence.
    
    Args:
        sentence (str): Input sentence.
        lambda_value (float): Lambda parameter for Poisson distribution.
    
    Returns:
        str: Noised sentence.
    """

    words = sentence.split()

    wl = len(words)

    if wl == 0:
        return ""
    
    mask_length = min(wl, poisson(lam=lambda_value))

    if mask_length == wl:
        mask_start = 0
    else:
        mask_start = random.choice(range(wl - mask_length))

    words = ["<mask>" if i >= mask_start and i < mask_start + mask_length else words[i] for i in range(len(words))]
    # del words[mask_start:mask_start+mask_length]
    # words.insert(mask_start, "_")
   
    return " ".join(words)