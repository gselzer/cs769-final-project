from typing import List
import stanza

class POSTagger():
    def __init__(self, lang: str = "en"):
        stanza.download(lang)
        self.nlp = stanza.Pipeline(lang, processors='tokenize,pos', tokenize_no_ssplit=True)
    
    def tag(self, sentences: List[str]):
        doc = self.nlp("\n\n".join(sentences))
        return [" ".join([f"{d.text} {d.pos}" for d in s.words]) for s in doc.sentences]

def tag(file_name: str, lang: str):
    with open(file_name) as f:
        lines = f.readlines()
    tagger = POSTagger(lang)
    new_lines = tagger.tag(lines)
    with open(file_name, "w") as f:
        f.write("\n".join(new_lines))
