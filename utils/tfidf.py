import numpy as np
import pathlib
import os
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.pipeline import Sentencizer
from utils import utils

def rank(corpus, glossary, frame_keywords, length, criterion):
    """
    Ranks the sentences in the corpus according to their cosine similarity to the generic climate topic vector ("glossary").
    Only keeps the top n portion of the sentences (according to the "length" parameter).

    Arguments:
    corpus          — the corpus
    glossary        — list of climate change words
    frame_keywords  — frame-specific keywords extracted using tf-idf
    length          — the length to reduce the article to
    criterion       — ranking criterion: cosine
    """

    # Set up the name of target directory
    parent_dir = pathlib.Path(__file__).parents[1]
    if frame_keywords != None:
        target_dir = parent_dir.joinpath("results/tfidf-top-{}-percent-with-frame-embeddings".format(int(length*100)))
    elif frame_keywords == None:
        target_dir = parent_dir.joinpath("results/glove-top-{}-percent".format(int(length*100)))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
