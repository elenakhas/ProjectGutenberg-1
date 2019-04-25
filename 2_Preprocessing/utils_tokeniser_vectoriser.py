import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet 
from nltk.tokenize import wordpunct_tokenize
import collections
from collections import Counter
import re, string, math

wnl = nltk.WordNetLemmatizer()


def _tokenise_documents(texts_list):
    '''
    Takes a flat list (each containing strings - corresponding to a sentence or document), tokenises each sentence 
    using NLTK's word_tokenize and stores the results in a list. Does this for each sentence and then returns a 
    list of lists. 
    '''
    _tokenised_text_list = []
    for text in texts_list: 
        _tokens = nltk.tokenize.word_tokenize(text)
        _tokenised_text_list.append(_tokens)

    return _tokenised_text_list


def documents_preprocessor(texts_list):
    '''
    Takes a text (a sentence, or a document) and preprocesses for the purposes of generating machine learning data from the 
    input. The preprocessing includes: (a) tokenisation, (b) removal of punctuation, (c) lemmatisation and lowercasing. 
    Returns a list of tokens from the input text. Calls on the text_preprocessor function. 
    '''
    _lemmatised_text_list = []
    
    for text in texts_list: 
        _tokens = text_preprocessor(text)
        __emmatised_text_list.append(_tokens)
    
    return _lemmatised_text_list 

def make_ngrams (processed_texts_lists, n_gram=2, add_padding=False):
    '''
    Takes a list of lists (the lists contain pre-processed tokens) and produces n-gram tokens. Outputs a list of lists (in
    the same structure as the input). 
    '''
    # empty list to store the generated n-grams 
    ngrams_list = [] 
    
    
    # a for-loop just to iterate the number of times equal to the num of tokens in list 
    for processed_list in processed_texts_lists:
        _list = []
        counter = 0 
        for token in processed_list:
            # try-except to handle IndexErrors 
            try:
                ngram = " ".join([processed_list[0+i] for i in range(counter,counter+n_gram)])

            except IndexError: 
                if add_padding==False: 
                    break
                elif add_padding==True: 
                    # grab the remaining words that have not had n-grams generated from each of their positions
                    remaining_words = [processed_list[-1-i] for i in range(len(processed_list)-counter)]
                    # reverse the list since it was adding from the end of the previous 
                    remaining_words.reverse()
                    # end "<END>" tokens to pad the remaining spaces in the n-gram
                    ngram = " ".join(remaining_words + ["<END>"]*(n_gram-len(remaining_words)))

            _list.append(ngram)
            counter+=1
        ngrams_list.append(_list)
    return ngrams_list

def compute_global_vocabulary(text_lists):
    '''
    takes a list containing texts, (i) processes it and (ii) returns a pandas dataframe with the count of each n-word. 
    The processing involves: (a) tokenisation, (b) removal of punctuation, (c) lemmatisation and lowercasing. 
    '''
    _vocab_df = pd.DataFrame()
    counter = 0
    for text in text_lists: 
        _processed_text = text_preprocessor(text)
        _counter = collections.Counter(_processed_text)
        _text_df = pd.DataFrame([list(_counter.values())], columns = list(_counter.keys()))
        
        _vocab_df = pd.concat([_vocab_df, _text_df], sort=False, ignore_index=True)
        counter +=1
        if counter%250 ==0:
            print (counter)
    
    return _vocab_df 


def filter_df_shortsent(dataframe, min_tokens):
    __tokeep_index = [    ]
    [__tokeep_index.append(i) for i in dataframe.index if dataframe.loc[i].sum() >= min_tokens]
    return dataframe.loc[__tokeep_index]


def compute_vocab_freq(dataframe, top_n = 10, ascending=False):
    return dataframe.describe().loc['count'].sort_values(ascending=ascending)[0:top_n]


def compute_frequency (data_lists , add_normalisation = None):
    '''
    takes a list of lists (each comprising processed inputs for a particular sentence or document). 
    
    input | data:list - a list of lists. Each list inside contains either tokens, lemmas, bigrams, or trigrams. 
    
    add_normalisation = "tf_max" implements max tf normalisation 
    (see https://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html)
    add_normalisation = "relative_frequency"
    '''
    dictcount_lists = []
    
    def dictcount_maker(row):
        '''
        helper function to take a list of tokens (that comprise/make up a sentence) and generates a dictionary with
        the count of each token. 
        '''
        dictcount = {}
        for token in row:
            try: 
                dictcount[token] += 1
            except: 
                dictcount[token] = 1
        return dictcount
    
    if add_normalisation == None: 
        for row in data_lists:
            dictcount = dictcount_maker(row)
            dictcount_lists.append(dictcount)

            
    elif add_normalisation == "tf_max":
        for row in data_lists:
            dictcount = dictcount_maker(row)
            
            # get the max count within the dictionary  
            maxcount=max(dictcount.values())
            
            # dictionary comprehension to divide each count in the dictionary by the max count 
            # and apply a weight and "bias" to get the normalised frequency. 
            dictcount = {key:0.4+(1-0.4)*(count/maxcount) for key,count in dictcount.items()}
            
            dictcount_lists.append(dictcount)

            
    elif add_normalisation ==  "relative_frequency":
        for row in data_lists:
            dictcount = dictcount_maker(row)
            
            # get the sum of all frequency counts for each token. 
            totalcount=sum(dictcount.values())
            
            # dictionary comprehension to divide each count in the dictionary by the total count 
            # to get the relative frequency. 
            dictcount = {key:count/totalcount for key,count in dictcount.items()}
            
            dictcount_lists.append(dictcount)
    
    else: 
        print("The add_normalisation parameter chosen is not recognised. Please check.")
    
    return dictcount_lists

def vectoriser(dictcount_lists):
    '''
    
    '''
    # pseudocode: 
    # 1. iterate through the list of dictcounts. get all keys and add to a set (call this vocab). set because it is sorted 
    # and no repeated values
    # 2. iterate through all the list of distcounts again. create an empty np array of the same size as the vocab. get index 
    # of each key in the distcount in the vocab. add values of distcount to the respective entry in the nparray. 

    # question: what is the impact of the choice tf_max and relative_frequency? i.e. values in the np are no longer zeros and
    # and counts, but zeros and frequencies (at sentence/document level)
    
    vocabulary_set = set()
    for distcount in dictcount_lists:
        sentence_tokens = set(distcount.keys())
        vocabulary_set.update(sentence_tokens)
    
    # convert the vocab into a list so that we can use its index 
    vocabulary_list = list(vocabulary_set)
    
    # 
    vectorised_arrays = []
    for distcount in dictcount_lists:
        __array = np.zeros(len(vocabulary_list))
        for token in distcount: 
            index_in_vocab = vocabulary_list.index(token)
            __array[index_in_vocab] = distcount[token]
            
        vectorised_arrays.append(__array)
        
        
    return vectorised_arrays, vocabulary_list


def filterfunction(vectorised_arrays, vocabulary_list, datamin_freq = 10):
    '''
    takes a list of vectorised arrays as well as its associated vocabulary list. checks the global count/frequency 
    (normalised or not), removes from all arrays the columns where the global count/frequency (normalised or not)
    is below the provided value. removes the same columns from the vocabulary list returns (i) a new list of arrays;
    and (ii) a new vocabulary list.
    '''
    # pseudocode
    # easiest way: place all vectorised_arrays into a pandas and sum values of each column. then filter out 
    # manual way: concat all np.arrays. slice by column. sum it out and delete if < certain value. remember to remove 
    # the corresponding word in the vocab list. 
    
    # since each np array is the same size and columns are all aligned (i.e. indexed to the vocab list), we can just 
    # sum all the np arrays, this will generate a 1D array with the sum on each of the columns
    total_freqs = sum(vectorised_arrays)
    
    # identify the columns for words that have counts less than datamin_freq
    to_delete = []
    for col, val in enumerate(total_freqs):
        if val < datamin_freq:
            to_delete.append(col)
    
    # remove the columns from each of the vectorised arrays 
    new_vectorised_arrays = []
    for vectorised_array in vectorised_arrays:
        new_vectorised_arrays.append(np.delete(vectorised_array, to_delete))
    
    # remove the columns from vocabulary list 
    new_vocabulary_list = np.delete(vocabulary_list, to_delete)
    
    return new_vectorised_arrays, new_vocabulary_list


def text_preprocessor(text):
    '''
    Takes a text (a sentence, or a document) and preprocesses for the purposes of generating machine learning data from the 
    input. The preprocessing includes: (a) tokenisation, (b) removal of punctuation, (c) lemmatisation and lowercasing. 
    Returns a list of tokens from the input text. 
    '''
    _processed = []
    
    # tokenize the string
    _tokens = word_tokenize(text)
    # use nltk's pos_tag function to get the pos_tag for the string of tokens. 
    _tokens_postags = pos_tag(__tokens)

    
    for token_postag in _tokens_postags:  
        if token_postag[1] not in string.punctuation:
        # use get_wordnet_pos helper function to get the equivalent WordNetLemmatiser pos-tag
            wn_pos = get_wordnet_pos(token_postag)
            # WordNetLemmatiser only has tags for a, n, v, r. if-else to handle this. 
            if wn_pos != None: 
                _lemma = wnl.lemmatize(token_postag[0], wn_pos).lower()
            else:
                _lemma = token_postag[0].lower()
            _processed.append(_lemma)
    
    return _processed

def _get_wordnet_pos(word_pos_tuple):
    """
    Helper function for text_preprocessor. Takes a tuple of (token, pos_tag) generated from running a tokenised 
    sentence through nltk.word_tokenize, and maps POS tag to the first character that nltk wordnetlemmatizer's 
    .lemmatize() method accepts
    source: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#wordnetlemmatizerwithappropriatepostag 
    """
    tag = word_pos_tuple[1][0]
    tag_dict = {"J": nltk.wordnet.ADJ,
                "N": nltk.wordnet.NOUN,
                "V": nltk.wordnet.VERB,
                "R": nltk.wordnet.ADV}

    return tag_dict.get(tag)
