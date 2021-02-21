import argparse
import os
import pickle
from utils import utils, glove

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo for frame classifier.')
    parser.add_argument('--glossary', help='Type of climate change glossary to use. Options: normal, enriched', default="normal")
    parser.add_argument('--length', help='Length of how much text to keep. In range [0,1]', type=float, default=0.5)
    parser.add_argument('--method', help='The method for ranking of sentences. Options: glove, tfidf')
    parser.add_argument('--criterion', help='The criterion used for ranking. Options: cosine, wmd', default='cosine')
    parser.add_argument('--frame', help='Use frame-specific information for ranking, (0 or 1)', type=int, default=0)

    # Retrieve the arguments and make sure they're within in the acceptable range of vaues
    args = parser.parse_args()

    glossary = args.glossary

    length = args.length
    assert (length > 0 and length<=1), "\"length\" argument must be in the (0,1] range"

    method = args.method

    criterion = args.criterion
    if criterion == "wmd":
        assert method == "glove", "Word Mover's Distance can only be used with GloVe."

    frame = args.frame
    if frame == 1: # If frame specific ranking is selected, load the pre-computed frame-embeddings
        assert os.path.isfile("data/resources/frame_embeddings_dict.pkl"), "frame_embeddings_dict.pkl file has not yet been created. Make sure you run utils/extract_frame_keywords.py first."
        assert os.path.isfile("data/resources/frame_keywords_dict.pkl"), "frame_keywords_dict.pkl file has not yet been created. Make sure you run utils/extract_frame_keywords.py first."
        print("Loading frame keywords and embeddings...")
        with open("data/resources/frame_embeddings_dict.pkl", "rb") as file:
            frame_embeddings = pickle.load(file)
        with open("data/resources/frame_keywords_dict.pkl", "rb") as file:
            frame_keywords = pickle.load(file)
    elif frame == 0:
        frame_embeddings = None
        frame_keywords = None

    # Load the corpus and the chosen climate change glossary
    corpus = utils.load_corpus() # corpus is a dictionary with "ScienceOCR" and "NatureOCR" keys
    cc_words = utils.load_cc_words(glossary)

    # Call the corresponding function
    if method == "glove":
        if criterion == "cosine":
            glove.rank(corpus, cc_words, frame_embeddings, length, criterion)
        elif criterion == "wmd":
            glove.rank(corpus, cc_words, frame_keywords, length, criterion)
    if method == "tfidf":
        tfidft.rank()
        pass # call tfidf and implement other flags for tfidf
