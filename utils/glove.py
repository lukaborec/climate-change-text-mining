import numpy as np
import pathlib
import os
from scipy.spatial import distance
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.pipeline import Sentencizer
from sentence_transformers import SentenceTransformer
from utils import utils

def rank(corpus, glossary, frame_keywords, length=0.5):
    """
    Ranks the sentences in the corpus according to their cosine similarity to the generic climate topic vector ("glossary").
    Only keeps the top n portion of the sentences (according to the "length" parameter).
    """
    assert (length > 0 and length<=1), "length argument must be in the (0,1) range"

    # target directory name
    parent_dir = pathlib.Path(__file__).parents[1]
    if frame_keywords != None:
        target_dir = parent_dir.joinpath("results/glove-top-{}-percent-with-frame-embeddings".format(int(length*100)))
    elif frame_keywords == None:
        target_dir = parent_dir.joinpath("results/glove-top-{}-percent".format(int(length*100)))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # spaCy stuff
    nlp = English()
    nlp.add_pipe("sentencizer")
    tokenizer = Tokenizer(nlp.vocab)

    # initialize GloVe embeddings
    print("Loading GloVe embeddings...")
    glove_embeddings = SentenceTransformer('average_word_embeddings_glove.6B.300d')

    # create a generic climate change embedding
    cc_embedding = glove_embeddings.encode(glossary)
    cc_embedding = np.mean(cc_embedding, axis=0)

    # load master master_table
    table = utils.load_master_table()

    print("Processing articles...")
    for key in corpus.keys():
        # create a folder for ScienceOCR and NatureOCR
        target_dir_folder = target_dir.joinpath("{}".format(key))
        if not os.path.exists(target_dir_folder):
            os.makedirs(target_dir_folder)

        for article in corpus[key]:
            if frame_keywords != None:
                # find the article's framing in the master table
                art = article[0].replace(".ocr", "")
                try: # This is needed because some articles do not have a framing in the master table
                    category = table[table['txt']==art]["Labels_dominant"].values[0]
                    frame_embedding = frame_keywords[category]
                except: # In case they dont, their framing defaults to cc glossary
                    frame_embedding = glove_embeddings.encode(glossary)
                    frame_embedding = np.mean(frame_embedding, axis=0)

            # split article into sentences
            sentences = []
            doc = nlp(article[1])
            for sent in doc.sents:
                sentences.append(sent.orth_) # orth_ saves sentence as a string rather than a spacy doc object

            keep_length = int(len(sentences)*length) # keep only the best ranking portion of the article (portion expressed in "length, e.g. 0.5)

            # rank sentences
            scores = []
            for sent in sentences:
                sent_embedding = glove_embeddings.encode([token.orth_ for token in tokenizer(sent)])
                sent_embedding = np.mean(sent_embedding, axis=0)
                if frame_keywords != None: # for frame-specific-ranking
                    scores.append((sent, distance.cosine(frame_embedding, sent_embedding)))
                else: # for ranking according to climate change vector
                    scores.append((sent, distance.cosine(cc_embedding, sent_embedding)))
            # write reduced article to disk
            write_file = target_dir_folder.joinpath("{}".format(article[0]))
            with open(write_file, "a+") as file:
                for sent in sorted(scores, key=lambda x: x[1], reverse=0)[:keep_length]:
                    s = sent[0].strip() + " "
                    file.write(s)
    print("Done!")
