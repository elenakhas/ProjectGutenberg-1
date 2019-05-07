import spacy
import re
import nltk
from collections import Counter
import glob
import os
from spacy.pipeline import Sentencizer
from nltk.parse.stanford import StanfordParser
from spacy.lang.en.stop_words import STOP_WORDS
# load spacy stopwords to be used in wordcloud
stopwords = STOP_WORDS

# load the English model for spacy processing (we are using the most lightweigh one), disable the dependency parser
# to decrease the loading speed as we are going to use CoreNLP Stanford parser instead
nlp = spacy.load('en_core_web_sm', disable=["parser"])
sentencizer = Sentencizer(punct_chars=[".", "?", "!", "..."])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

# Store custom stop tokens to remove in the processing
# for instance, due to encoding inconsistency in the original Project Gutenberg data, some tokens can not be filtered
# out by spaCy (e.g. this tabulation character is not recognized as tabulation, but gets a "NOUN" postag)
to_remove = ["\t", "      "]

#Load Stanford parser from CoreNLP for syntactic parsing
java_path = r'/usr/lib/jvm/java-8-oracle/jre/bin/java'
os.environ['JAVAHOME'] = java_path
scp = StanfordParser(path_to_jar='stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar',
           path_to_models_jar='stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2-models.jar')


def create_a_doc(filenames):
    '''Creates a joint spaCy document from a list of files
    Inputs: filename (path to the file to process)
    Outputs: an annotated spaCy document
    '''
    allfiles = ""
    for filename in filenames:
        with open(filename, "r", encoding='utf-8') as fr:
            file = fr.read()
            file = re.sub(r'([-]{2,}|[“_]|(\*\s)+)', "", file)
            allfiles = allfiles + "\t" + file
    allfiles = nlp(allfiles.lstrip("\t"))
    return allfiles

def string_to_doc(string):
    '''Converts a string into spacy processed document'''
    return nlp(string)

def process_an_author(author_id, readpath = "./data/booksample_txt/"):
    '''
    Reads files from a directory for one author, creates a joint spacy document
    Inputs: path to the file to read; author_id
    Returns: annotated spacy document for one author
    '''
    files = []
    for filename in glob.iglob(readpath + "*.txt"):
        # get the author_id from the filename
        res = re.findall(r'\\(.+)', filename)[0].split("_")
        # create a spaCy document for the file
        if res[0] == author_id:
            files.append(filename)
#     collect the books from the df
    doc = create_a_doc(files)
    return doc

def segment_sentences(doc):
    '''Returns the sentences segmented by spacy rule-based sentence segmenter
    Inputs: doc - spaCy document
    Outputs: list of strings containing sentences
    '''
    return [sent for sent in doc.sents]


def get_book_word_tokens(doc):
    '''Returns tokens in a file
    Inputs: doc - spaCy document
    Outputs: list of strings - lowercased tokens; with removed punctuation
    and custom undesired tokens
    '''
    return [token.text.lower() for token in doc if not token.is_punct and not token.text in to_remove]

def postagging(doc):
    '''Performs postagging on a file
    Inputs: spaCy annotated document
    Outputs: a dictionary with keys - postags, values - lists of tokens corresponding to a postag
    Uses WordNet postags: https://spacy.io/api/annotation#pos-tagging
    '''
    tags = ["NOUN", 'VERB', 'ADJ', "ADV", "AUX", "INTJ", "NUM", "PRON", "PROPN", "PUNCT"]
    other = ["ADP", 'CONJ', "CCONJ", "DET", "PART", "SCONJ"]
    all_tags = tags + other
    postags = dict()
    other_pos = []
    for tag in all_tags:
        if tag in other:
            other_pos.extend([token.text for token in doc if token.pos_ == tag])
            postags["OTHER"] = other_pos
        else:
            this_tag = [token.text for token in doc if token.pos_ == tag]
            postags[tag] = this_tag

    return postags


def ne_extraction(doc):
    ''' Extracts named entities of place, person, date types
    Inputs: spaCy annotated document
    Outputs: a dictionary witk keys = NE types, values = corresponding tokens
    '''
    entities = dict()
    places = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
    entities["places"] = places
    entities["persons"] = persons
    entities["dates"] = dates
    return entities


def lemmatization(doc, tag=None):
    '''
    Lemmatizes the tokens of a specific POS or all tokens
    Inputs: spaCy annotated document, optional - tag, a specified POS tag
    The default value None would process all the tokens
    Outputs: a list of lemmas for the selected tokens
    '''
    if tag == None:
        lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.text in to_remove]
    else:
        lemmas = [token.lemma_ for token in doc if
                  token.pos_ == tag and not token.is_punct and not token.text in to_remove]
    return lemmas


def const_parsing(sentences, n):
    '''Parses a sentence using Stanford CoreNLP constituency parsing
    Inputs: a list of sentences
    Outputs: a list of parse trees for all the sentences in a list
    '''
    trees = []
    for sentence in sentences[:n]:
        sentence = sentence.text
        parse_trees = list(scp.raw_parse(sentence))
        tree = parse_trees[0]
        trees.append(tree)
    return trees


def get_sub_trees(trees, tag="NP", n=10):
    '''Extracts subtrees of a specified type (tag)
    Inputs: trees - a list of parse trees; tag - a specified phrase type to extract
    (NP, VP, DP, etc.)
    Outputs: a list of subtrees of a specific type
    '''
    labeled_nodes = []
    for tree in trees[:n]:
        one_sent = []
        for s in tree.subtrees(lambda tree: tree.label() == tag):
            one_sent.append((s.leaves(), s.label()))
            labeled_nodes.append(one_sent)
    return labeled_nodes
