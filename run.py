import argparse
import os
import pickle
from utils import utils, glove

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo for frame classifier.')
    parser.add_argument('--glossary', help='Type of climate change glossary to use: normal, enriched', default="normal")
    parser.add_argument('--length', help='Length of how much text to keep. In range [0,1]', type=float, default=0.5)
    parser.add_argument('--method', help='The method for ranking of sentences Options: wmd, glove, w2v, tfidf')
    parser.add_argument('--frame', help='Use frame-specific information for ranking, (0 or 1)', default=0)
    parser.add_argument('--ranking', help="Measurement to use for ranking: cosine, ")
    # if --frame=1
    pass
    # if --method=tfidf
    pass
    # Retrieve the arguments
    args = parser.parse_args()
    glossary = args.glossary
    length = args.length
    method = args.method
    frame = args.frame
    if frame == '1':
        assert os.path.isfile("data/resources/frame_keywords_dict.pkl"), "frame_keywords_dict.pkl file has not yet been created. Make sure you run utils/extract_frame_keywords.py first."
        print("Loading frame keywords...")
        with open("data/resources/frame_keywords_dict.pkl", "rb") as file:
            frame_keywords = pickle.load(file)
    elif frame == '0':
        frame_keywords = None

    corpus = utils.load_corpus() # corpus is a list of (article_name, article_text) tuples
    cc_words = utils.load_cc_words(glossary)


    if method == "glove":
        glove.rank(corpus, cc_words, frame_keywords, length)
    if method == "tfidf":
        pass # call tfidf and implement other flags for tfidf
