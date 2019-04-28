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

def _filterdf_shortsents(dataframe, min_tokens, pre_computed = True, col_name = "sent_length"):
    '''
    takes a dataframe with preprocessed information about a sentence or a document (multiple sentences)
    and returns a dataframe with only instances that meet the min_tokens requirements. 
    input | either of: (i) a df with a pre_computed sentence length, or (ii) a dataframe with a list of
    tokens already produced for each of the instances. 
    '''
    if pre_computed == True: 
        return dataframe[dataframe[col_name]>=min_tokens]
    else: 
        _tokeep_index = [    ]
        [_tokeep_index.append(i) for i in dataframe.index if len(dataframe.loc[i,col_name]) >= min_tokens]
        return dataframe.loc[_tokeep_index]
                  

def _compute_globalvocab(token_lists):
    '''
    takes a list containing texts, (i) processes it and (ii) returns a pandas dataframe with the count of each n-word. 
    The processing involves: (a) tokenisation, (b) removal of punctuation, (c) lemmatisation and lowercasing. 
    '''
    _globalvocab_df = pd.DataFrame()
    for token_list in token_lists: 
        _counter = collections.Counter(token_list)
        _text_df = pd.DataFrame([list(_counter.values())], columns = list(_counter.keys()))
        
        _vocab_df = pd.concat([_vocab_df, _text_df], sort=False, ignore_index=True)
        counter +=1
    
    return _vocab_df 

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

def _compute_globalvocabfreq(dataframe, top_n = 10, ascending=False):
    return dataframe.describe().loc['count'].sort_values(ascending=ascending)[0:top_n]



def _compute_frequency(data_lists , add_normalisation = None):
    '''
    takes a list of lists (each comprising processed inputs for a particular sentence or document). 
    input | data:list - a list of lists. Each list inside contains either tokens, lemmas, bigrams, or trigrams. 
    add_normalisation = "tf_max" implements max tf normalisation 
    (see https://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html)
    add_normalisation = "relative_frequency"
    '''
    freqdict_list = []
    
    
    if add_normalisation == None: 
        for row in data_lists:
            dictcount = _dictcount_maker(row)
            freqdict_list.append(dictcount)

            
    elif add_normalisation == "tf_max":
        for row in data_lists:
            dictcount = _dictcount_maker(row)
            
            # get the max count within the dictionary  
            maxcount=max(dictcount.values())
            
            # dictionary comprehension to divide each count in the dictionary by the max count 
            # and apply a weight and "bias" to get the normalised frequency. 
            dictcount = {key:0.4+(1-0.4)*(count/maxcount) for key,count in dictcount.items()}
            
            freqdict_list.append(dictcount)

            
    elif add_normalisation ==  "relative_frequency":
        for row in data_lists:
            dictcount = _dictcount_maker(row)
            
            # get the sum of all frequency counts for each token. 
            totalcount=sum(dictcount.values())
            
            # dictionary comprehension to divide each count in the dictionary by the total count 
            # to get the relative frequency. 
            dictcount = {key:count/totalcount for key,count in dictcount.items()}
            
            freqdict_list.append(dictcount)
    
    else: 
        print("The add_normalisation parameter chosen is not recognised. Please check.")
    
    return freqdict_list


def _dictcount_maker(row):
    '''
    helper function to take a list of tokens (that comprise/make up a sentence) and generates a dictionary with
    the count of each token. 
    '''
    dictcount = {}
    for token in row:
        try: # if key already exists
            dictcount[token] += 1
        except: # if key does not exist yet (i.e. new word seen)
            dictcount[token] = 1
    return dictcount


def _vectoriser(freqdict_list):
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
    for distcount in freqdict_list:
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
