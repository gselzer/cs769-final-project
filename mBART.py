from numpy.random import poisson
import random
from typing import List

def mbart(src_files: List[str], tgt_files: List[str], p: float = 0.35, lambda_value: float = 3.5):
    """
    Apply noising to each sentence in the input file and save the result to the output file.
    
    Args:
        infile (str): Input file path.
        outfile (str): Output file path.
        p (float): Probability of applying noising to each sentence.
        lambda_value (float): Lambda parameter for Poisson distribution.
    """
    for i in range(len(src_files)):

        with open(src_files[i], "r") as f:
            sentences = f.read().split("\n")
        
        noised_sentences = random.shuffle([noising(sentence, p, lambda_value) for sentence in sentences])
        
        with open(tgt_files[i], "w") as f:
            f.write("\n".join(noised_sentences))


def noising(sentence: str, p, lambda_value):
    """
    Apply mBART-style noising to a sentence.
    
    Args:
        sentence (str): Input sentence.
        p (float): Probability of applying each noising operation.
        lambda_value (float): Lambda parameter for Poisson distribution.
    
    Returns:
        str: Noised sentence.
    """
    words = sentence.split()

    # Masking 35% of the words
    # num_words_to_mask = int(len(words) * p)
    # mask_start_indices = sorted(random.sample(range(len(words)), num_words_to_mask))

    # for start_index in mask_start_indices:
    #     span_length = max(1, int(poisson(lambda_value)))
    #     end_index = min(len(words), start_index + span_length)
    #     words[start_index:end_index] = ["[MASK]"] * (end_index - start_index)


    wl = len(words)

    if wl == 0:
        return ""

    mask_length = int(wl * p)
    mask_start = random.choice(range(wl - mask_length))
    # mask_end = mask_start + mask_length
    
    # words[mask_start:mask_end] = ["[MASK]"] * mask_length

    for i in range(mask_length):
        words[mask_start+i] = "[MASK]"

    # # Randomly shuffle the order of words
    # random.shuffle(words)
    
    return " ".join(words)