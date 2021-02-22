import numpy as np
import pathlib
import os
import nltk
from nltk.corpus import stopwords
from scipy.spatial import distance
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.pipeline import Sentencizer
# from sentence_transformers import SentenceTransformer
import gensim.downloader
from gensim.models import KeyedVectors
import pyemd # For calculating WMD
from utils import utils

def rank(corpus, glossary, frame_keywords, length, criterion):
    """
    Ranks the sentences in the corpus according to their cosine similarity to the generic climate topic vector ("glossary").
    Only keeps the top n portion of the sentences (according to the "length" parameter).

    Arguments:
    corpus          — the corpus
    glossary        — list of climate change words
    frame_keywords  — frame-specific embeddings/keywords retrieved from tf-idf keyword extraction
    length          — the length to reduce the article to
    criterion       — ranking criterion: cosine, wmd
    """

    # Set up the name of target directory
    parent_dir = pathlib.Path(__file__).parents[1]
    if frame_keywords != None:
        target_dir = parent_dir.joinpath("results/glove-top-{}-percent-with-frame-embeddings-{}".format(int(length*100),criterion))
    elif frame_keywords == None:
        target_dir = parent_dir.joinpath("results/glove-top-{}-percent-{}".format(int(length*100),criterion))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # spaCy stuff
    nlp = English()
    nlp.add_pipe("sentencizer")
    tokenizer = Tokenizer(nlp.vocab)

    # Initialize GloVe embeddings
    print("Loading GloVe embeddings...")
    glove_embeddings = gensim.downloader.load("glove-wiki-gigaword-300")

    # Download stop cc_words
    stop_words = stopwords.words("english")

    # Create a generic climate change embedding
    cc_words = [word.lower() for word in glossary if word.lower() in glove_embeddings.vocab]
    cc_embedding = glove_embeddings[cc_words]
    cc_embedding = np.mean(cc_embedding, axis=0)

    # load master_table dataframe - this is used to retrieve article's framing
    table = utils.load_master_table()

    print("Processing articles...")
    for key in corpus.keys():
        # Create a folder for ScienceOCR and NatureOCR in target dir
        target_dir_folder = target_dir.joinpath("{}".format(key))
        if not os.path.exists(target_dir_folder):
            os.makedirs(target_dir_folder)

        # Query for the case when not using frame-specific information
        if criterion == "cosine":
            query = cc_embedding
        elif criterion == "wmd":
            query = cc_words

        # Loop over articles
        for article in corpus[key]:
            if frame_keywords != None: # i.e. if frame_keywords == 1
                # find the article's framing in the master table
                art = article[0].replace(".ocr", "")
                try: # This is needed because some articles have missing values for framing in the master table
                    category = table[table['txt']==art]["Labels_dominant"].values[0] # retrieves the frame
                    if criterion == "wmd":
                        query = [word.lower() for word in frame_keywords[category] if word.lower() in glove_embeddings.vocab] # retrieves the frame's embedding
                    elif criterion == "cosine":
                        query = frame_keywords[category]
                except: # In case an article doesnt have a framing in the master table, we use cc_embedding instead
                    if criterion == "cosine":
                        query = cc_embedding
                    elif criterion == "wmd":
                        query == cc_words

            # Split the article into sentences
            sentences = []
            doc = nlp(article[1])
            for sent in doc.sents:
                sentences.append(sent.orth_) # orth_ saves sentence as a string rather than a spacy doc object

            keep_length = int(len(sentences)*length) # keep only the best ranking portion of the article (portion expressed in "length, e.g. 0.5)

            # Rank sentences
            scores = []
            for sent in sentences:
                # Create sentence embedding
                sentence = [token.orth_ for token in tokenizer(sent)]
                sentence = [word.lower() for word in sentence if word.lower() in glove_embeddings.vocab and word.lower() not in stop_words]
                if len(sentence)==0:
                    continue
                if criterion == "cosine":
                    sent_embedding = glove_embeddings[sentence]
                    sent_embedding = np.mean(sent_embedding, axis=0)
                    scores.append((sent, distance.cosine(query, sent_embedding)))
                elif criterion == "wmd":
                    scores.append((sent, glove_embeddings.wmdistance(query, sentence)))

            # Write reduced article to disk
            write_file = target_dir_folder.joinpath("{}".format(article[0]))
            with open(write_file, "w") as file:
                for sent in sorted(scores, key=lambda x: x[1], reverse=0)[:keep_length]:
                    file.write(sent[0].strip() + " ")
    print("Done!")
