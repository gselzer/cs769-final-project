from numpy.random import poisson
import random
from typing import List

def mbart(src_files: List[str], tgt_files: List[str], lambda_value: float = 3.5):
    """
    Apply noising to each sentence in the input file and save the result to the output file.
    
    Args:
        infile (str): Input file path.
        outfile (str): Output file path.
        lambda_value (float): Lambda parameter for Poisson distribution.
    """
    for i in range(len(src_files)):

        with open(src_files[i], "r") as f:
            sentences = f.read().split("\n")
        
        noised_sentences = [noising(sentence, lambda_value) for sentence in sentences]
        random.shuffle(noised_sentences)

        print(noised_sentences)

        with open(tgt_files[i], "w") as f:
            f.write("\n".join(noised_sentences))


def noising(sentence: str, lambda_value):
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

    mask_length = min(wl, poisson(lam=lambda_value)) #int(wl * p)

    if mask_length == wl == 0:
        return ""
    
    mask_start = random.choice(range(wl - mask_length))

    del words[mask_start:mask_start+mask_length]
   
    return " ".join(words)