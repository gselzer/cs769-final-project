import re
import random
from typing import Tuple, List

def sample_source_and_target(source: str, target: str, n: int) -> Tuple[List[str], List[str]]:

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

    # Randomly sample from the xml-less lines in the files
    selected = random.sample([x for x in zip(xmlless_source, xmlless_target)], n)

    return tuple(zip(*selected))