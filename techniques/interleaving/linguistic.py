from typing import List
import stanza
import pickle
import shutil
import urllib.request
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import os
from zipfile import ZipFile

class POSTagger():
    def __init__(self, lang: str = "en"):
        stanza.download(lang)
        self.nlp = stanza.Pipeline(lang, processors='tokenize,pos', tokenize_no_ssplit=True)
    
    def tag(self, sentences: List[str]):
        doc = self.nlp("\n\n".join(sentences))
        return [" ".join([f"{d.text} {d.pos}" for d in s.words]) for s in doc.sentences]

def _tag_stanza(file_name: str, lang: str):
    with open(file_name) as f:
        lines = f.readlines()
    tagger = POSTagger(lang)
    new_lines = tagger.tag(lines)
    with open(file_name, "w") as f:
        f.write("\n".join(new_lines))

def _featurize(words: List[str]) -> List[str]:
    output = []
    l = len(words)
    for i in range(l):
        data = {}
        data['prev-prev-token'] = words[i-2] if i-2 >= 0 else ""
        data['prev-token'] = words[i-1] if i-1 >= 0 else ""
        data['token'] = words[i]
        data['next-token'] = words[i+1] if i + 1 < l else ""
        data['next-next-token'] = words[i+2] if i + 2 < l else ""
        output.append(data)
    return output


def _get_classifier(lang: str):
    """
    Inspiration drawn from https://github.com/raju-shrestha/POS-Tagger-Nepali/tree/master,
    written by Raju Shrestha
    """
    current_dir = os.path.dirname(__file__)
    filename = os.path.join(current_dir, f"pos_tagger_{lang}.pickle")
    if os.path.exists(filename):
        return pickle.load(open(filename, "rb"))
    
    urllib.request.urlretrieve("http://www.cle.org.pk/Downloads/ling_resources/parallelcorpus/NepaliTaggedCorpus.zip", "NepaliTaggedCorpus.zip")
    tmpfolder = "nepali-tagged-data"
    if os.path.exists(tmpfolder):
        shutil.rmtree(tmpfolder)
    os.mkdir(tmpfolder)
    with ZipFile("NepaliTaggedCorpus.zip", "r") as f:
        f.extractall(tmpfolder)
    os.remove("NepaliTaggedCorpus.zip")

    # Train the tagger on the first file
    train_file = os.path.join(tmpfolder, "new_submissions_parallel_corpus_project_Nepal", "00ne_pos.txt")
    with open(train_file, "r") as f:
        train_lines = f.readlines()
    
    X_train = []
    y_train = []
    for line in train_lines:
        tokens = line.replace("<", " <").replace(">", "> ").split()
        if tokens == []:
            continue
        words = tokens[::2]
        tags = tokens[1::2]
        if len(words) != len(tags):
            continue
        if not all(["<" in t for t in tags]):
            continue
        if any(["<" in w for w in words]):
            continue
        X_train.extend(_featurize(words))
        y_train.extend(tags)

    clf = Pipeline([
                ('vectorizer', DictVectorizer(sparse=False)),
                ('classifier', LinearSVC())
            ])
    clf.fit(X_train[:50000],y_train[:50000])

    # # Test the tagger on the second file - OPTIONAL
    # test_file = os.path.join(tmpfolder, "new_submissions_parallel_corpus_project_Nepal", "01ne_pos.txt")
    # with open(test_file, "r") as f:
    #     test_lines = f.readlines()
    
    # X_test = []
    # y_test = []
    # for line in test_lines:
    #     tokens = line.replace("<", " <").replace(">", "> ").split()
    #     words = tokens[::2]
    #     tags = tokens[1::2]
    #     if len(words) != len(tags):
    #         continue
    #     if not all(["<" in t for t in tags]):
    #         continue
    #     if any(["<" in w for w in words]):
    #         continue
    #     l = len(words)
    #     X_train.extend(_featurize(words))
    #     y_test.extend(tags)
    
    # print(f"Trained Nepali POS Tagger with accuracy: {clf.score(X_test, y_test)}")
    shutil.rmtree(tmpfolder)
    pickle.dump(clf, open(filename, "wb"))

    return clf

def _tag_sklearn(file_name: str, lang: str):
    clf = _get_classifier(lang)

    with open(file_name, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        new_line = ""
        tokens = line.split()
        if tokens == []:
            new_lines.append(line)
            continue
        features = _featurize(line.split())
        tags = clf.predict(features)
        for token, tag in zip(tokens, tags):
            new_line += token + " " + tag + " "
        new_lines.append(new_line)

    with open(file_name, "w") as f:
        f.write("\n".join(new_lines))

def tag(file_name: str, lang: str):
    if lang in ["de", "en"]:
        _tag_stanza(file_name, lang)
    elif lang in ["ne"]:
        _tag_sklearn(file_name, lang)