import random
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.pipeline import Sentencizer
import os
from utils import utils

# spaCy stuff
nlp = English()
nlp.add_pipe("sentencizer")
tokenizer = Tokenizer(nlp.vocab)

corpus = utils.load_corpus()

current_dir = os.getcwd()
target_dir = os.path.join(current_dir, "results/random_ranking")
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for key in corpus.keys(): # The keys are ScienceOCR and NatureOCR
    # Create folders for ScienceOCR and NatureOCR in target dir where reduced articles will be saved
    target_dir_folder = os.path.join(target_dir, key)
    if not os.path.exists(target_dir_folder):
        os.makedirs(target_dir_folder)

    # Loop over articles in NatureOCR and ScienceOCR folders
    for article in corpus[key]:
        sentences = []
        # Split article into sentences
        doc = nlp(article[1])
        for sent in doc.sents:
            sentences.append(sent.orth_) # orth_ saves sentence as a string rather than a spacy doc object

        # Randomly sample indices that correspond to a half of the article
        keep_length = int(len(sentences) * 0.67)
        keep_indices = random.sample(list(range(len(sentences))), keep_length)

        # Add the sampled sentences to a list
        keep_sentences = []
        for index in keep_indices:
            keep_sentences.append(sentences[index])

        write_file = os.path.join(target_dir_folder, article[0])
        with open(write_file, "w") as file:
            for sentence in keep_sentences:
                file.write(sentence.strip() + " ")

print("Done!")
