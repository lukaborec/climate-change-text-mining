import os
import pathlib
import numpy as np
import pandas as pd
import gensim.downloader
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.pipeline import Sentencizer
from scipy.spatial import distance
import utils

def get_frame_keywords():
    # initialize stuff
    lemmatizer = WordNetLemmatizer()
    nlp = English()
    nlp.add_pipe("sentencizer")
    tokenizer = Tokenizer(nlp.vocab)
    print("Loading word2vec embeddings...")
    word2vec = gensim.downloader.load("word2vec-google-news-300")
    # get parent and resource directory for loading and saving stuff
    parent_dir = pathlib.Path(__file__).parents[1]
    resource_dir = parent_dir.joinpath("data/resources")

    # load master table
    table = utils.load_master_table()
    frames = table['Labels_dominant'].unique()
    frame_dict = {}
    for  f in frames:
        frame_dict[f]=[]
    # SORTS ARTICLES INTO FRAME GROUPS
    # e.g. frame_dict['SCIENTIFIC'] =  [a list of all the articles framed as scientific challenge]
    corpus_dir = parent_dir.joinpath("data")
    # Loop through folders ("ScienceOCR" and "NatureOCR")
    for file in os.listdir(corpus_dir):
        if file in ["NatureOCR", "ScienceOCR"]:
            path = os.path.join(corpus_dir, file)
            file_list = os.listdir(path)
            for article in file_list:
                art = article.replace(".ocr", "") # remove ".ocr" from name so the file can be looked up in the master table
                txt = open(os.path.join(path, article), "r", errors='ignore').readlines()
                if file == 'ScienceOCR':
                    txt = txt[3:] # Remove author name for Science articles
                else:
                    txt = txt[2:]
                txt = " ".join(str(sent.strip()) for sent in txt)
                try:
                    frame_dict[table[table['txt']==art]["Labels_dominant"].values[0]].append(txt)
                except IndexError:
                    pass
    # tfidf keyword extraction
    frame_keywords = {}
    for frame in frame_dict.keys():
        frame_keywords[frame] = [] # creates an empty list for each frame - keywords are later appended to this list
        texts = frame_dict[frame] # retrieves all texts belonging to a frame

        tfidf = TfidfVectorizer(stop_words="english")
        doc_term_matrix = tfidf.fit_transform(texts)
        words = np.array(tfidf.get_feature_names()) # Get the list of words

        for row in doc_term_matrix:
            l = list(enumerate(row.toarray()[0])) # Add indices to word scores to preserve the order when sorting word scores
            indices = [e[0] for e in sorted(l, key=lambda x:x[1], reverse=True)[:8]] # Get n highest ranking indices per article
            keywords = [words[i] for i in indices] # Extract keywords
            for word in keywords: # Add them to the corresponding frame keyword list
                frame_keywords[frame].append(word)

    # since lengths of articles vary, some framings have only ~50 keywords whereas some have many hundreds
    # the code below preprocesses the lists (lemmatizing and removing duplicates) and only keeps the top scoring 40 words per frame
    for key, values in frame_keywords.items():
        frame_keywords[key] = list(set(lemmatizer.lemmatize(word) for word in frame_keywords[key])) # Lemmatizing
        frame_keywords[key]  = [w[0] for w in Counter(frame_keywords[key]).most_common(40)] # Setting the keywords to be the most common 40 words

    # create mean embeddings for each frame
    cc_words = utils.load_cc_words()
    frame_embeddings = {}
    for frame, keywords in frame_keywords.items():
        frame_keywords[frame] += cc_words # add cc words to each framing - should this be done?
        kwords = [word for word in keywords if word in word2vec.vocab]
        embeddings = word2vec[kwords]
        embeddings = np.mean(embeddings, axis=0)
        frame_embeddings[frame] = embeddings

    # save embeddings to data/resources
    with open(os.path.join(resource_dir, "{}.pkl".format("frame_keywords_dict")), "wb") as file:
        pickle.dump(frame_embeddings, file)
    print("Frame keywords saved to data/resources!")

if __name__ == "__main__":
    get_frame_keywords()
