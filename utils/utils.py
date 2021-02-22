import pandas as pd
import numpy as np
import os
import pathlib
import argparse

def clean(cc_list):
    '''
    Cleans the list of climate change words.
    '''
    cc_list = [w.replace("\n", "") for w in cc_list]
    cc_list = [w.replace(" ", "_") for w in cc_list]
    return cc_list

def load_cc_words(type="normal"):
    '''
    Loads the glosssaries of climate change words.
    '''
    if type == "normal":
        cc_words = open("data/resources/CCglossaryWiki.txt", "r").readlines()
        cc_words = clean(cc_words)
        return cc_words
    elif type == "enriched":
        cc_words = pd.read_csv("data/resources/CCglossaryComplete.csv", index_col=False, names=['word'])['word'].values
        cc_words = clean(cc_words)
        return cc_words

def load_corpus():
    '''
    Loads the corpus and returns a a dictionary with "ScienceOCR" and "NatureOCR" as keys and articles (name_text) tuples as the key's element.
    '''
    # Get the one-level-up path which is where the "data" folder resides
    parent_dir = pathlib.Path(__file__).parents[1]
    corpus_dir = parent_dir.joinpath("data")

    corpus = {}
    corpus["ScienceOCR"] = []
    corpus["NatureOCR"] = []
    folder_list = os.listdir(corpus_dir)

    for folder in folder_list:
        if folder in ["ScienceOCR", "NatureOCR"]:
            folder_path = os.path.join(corpus_dir, folder) # Get folder path
            file_list = os.listdir(folder_path) # Get a list of files thart are in the folder
            for article in file_list:
                # article_name = article.replace(".ocr", "")
                article_path = os.path.join(folder_path, article)
                text = open(os.path.join(folder_path, article), "r", errors='ignore').readlines() # readlines() instead of read() because we want to remove the third sentence from Science articles because they contain the name of the author's name
                if folder == 'ScienceOCR':
                    text = text[3:] # Remove author name for Science articles
                else:
                    text = text[2:]
                text = " ".join(str(sent) for sent in text) # Add sentenes into one string
                corpus[folder].append((article, text))  # returns a list of (article_name, text) tuples
    return corpus                               # e.g. [(348181a0.txt, "blah blah), (...)]

def load_master_table():
    '''
    Loads the master table with annotations (the table is provided by Hulme et al).
    '''
    parent_dir = pathlib.Path(__file__).parents[1]
    resource_dir = parent_dir.joinpath("data/resources")
    table = pd.read_csv(os.path.join(resource_dir,"master_table.tsv"), delimiter="\t")
    RENAME_DICT = {"A_dominant_frame" : "ECON",
    "B_dominant_frame" : "DEVELOP",
    "C_dominant_frame" : "NATIONAL/INTERNATIONAL SECURITY",
    "D_dominant_frame" : "ETHICAL/MORAL",
    "E_dominant_frame" : "TECH",
    "F_dominant_frame" : "GOVERNANCE/INSITUTIONAL",
    "G_dominant_frame" : "SCIENTIFIC",
    "H_dominant_frame" : "COMMUNICATION"}
    table["Labels_dominant"] = table["Labels_dominant"].map(RENAME_DICT)
    return table
