import re
import random
from typing import Tuple, List

def read_source_and_target(source: str, target: str) -> Tuple[List[str], List[str]]:

    # This pattern will search for any tags
    tag_pattern = re.compile(r'<.*?>')

    # Read in the files
    with open(source) as f:
        text_source = f.read().split('\n')
    with open(target) as f:
        text_target = f.read().split('\n')

    # Filter out any strings that have a tag
    xmlless_source = [s for s in text_source if not re.search(tag_pattern, s)]
    xmlless_target = [s for s in text_target if not re.search(tag_pattern, s)]

    # Filter out any strings that have a tag
    xmlless_source = [s for s in text_source if not re.search(tag_pattern, s)]
    xmlless_target = [s for s in text_target if not re.search(tag_pattern, s)]

    selected = list(zip(xmlless_source, xmlless_target))

    # Filter out any lines where one of the pair is blank.
    # One time this happens is on line 78520 of the English input data.
    filtered = [s for s in selected if s[0] and s[1]]
    if len(selected) != len(filtered):
        print("Removed some blank lines")

    return tuple(zip(*filtered))