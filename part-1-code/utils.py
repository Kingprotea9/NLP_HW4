import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


import re
import random
from nltk.corpus import wordnet as wn
import nltk
try:
    wn.synsets("test")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
# small QWERTY neighbor set for gentle, plausible typos
_QWERTY_NEIGHBORS = {
    "q": "w", "w": "qe", "e": "wr", "r": "et", "t": "ry",
    "y": "tu", "u": "yi", "i": "uo", "o": "pi", "p": "o",
    "a": "qs", "s": "adw", "d": "sfe", "f": "dgr", "g": "fht",
    "h": "gjy", "j": "huk", "k": "jl", "l": "k",
    "z": "xs", "x": "zcd", "c": "xvf", "v": "cbg", "b": "vnh",
    "n": "bmj", "m": "n"
}

def _preserve_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src.istitle():
        return repl.capitalize()
    return repl

def _wordnet_synonym(w):
    from nltk.corpus import wordnet as wn
    w_low = w.lower()
    syns = wn.synsets(w_low)
    if not syns:
        return None
    # collect lemma candidates, drop identical strings, simple filter to avoid multiword lemmas
    cands = []
    for s in syns:
        for l in s.lemmas():
            name = l.name().replace("_", " ")
            if name.isalpha() and name.lower() != w_low:
                cands.append(name)
    if not cands:
        return None
    return random.choice(cands)

def _inject_mild_typo(w):
    # choose swap or neighbor insert; keep it very light
    if len(w) >= 2 and random.random() < 0.5:
        j = random.randrange(0, len(w) - 1)
        return w[:j] + w[j+1] + w[j] + w[j+2:]
    # neighbor insert
    j = random.randrange(0, len(w))
    ch = w[j].lower()
    neigh = _QWERTY_NEIGHBORS.get(ch)
    if not neigh:
        return w
    ins = random.choice(list(neigh))
    return w[:j] + ins + w[j:]

def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINS HERE ####

    text = example["text"]
    # split into words or single punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    p_syn = 0.25   # chance to synonym-replace a word
    p_typo = 0.15  # chance to add a mild typo when not synonym-replaced

    new_tokens = []
    for tok in tokens:
        if tok.isalpha():
            r = random.random()
            replaced = tok
            if r < p_syn:
                syn = _wordnet_synonym(tok)
                if syn:
                    replaced = _preserve_case(tok, syn)
            elif r < p_syn + p_typo:
                replaced = _inject_mild_typo(tok)
                # keep case after typo
                replaced = _preserve_case(tok, replaced)
            new_tokens.append(replaced)
        else:
            new_tokens.append(tok)

    # rebuild: add spaces before word-tokens, not before punctuation
    out = []
    for i, t in enumerate(new_tokens):
        if re.fullmatch(r"[^\w\s]", t):
            out.append(t)  # punctuation hugs previous token
        else:
            if i > 0 and not re.fullmatch(r"[^\w\s]", new_tokens[i-1]):
                out.append(" ")
            elif i > 0 and re.fullmatch(r"[^\w\s]", new_tokens[i-1]):
                out.append(" ")
            out.append(t)

    example["text"] = "".join(out).strip()

    ##### YOUR CODE ENDS HERE ######
    return example

