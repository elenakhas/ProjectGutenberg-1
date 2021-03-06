import spacy, re, glob, time, os
import pandas as pd 

# !pip install benepar[cpu]
import benepar
benepar.download('benepar_en2')
nlp = spacy.load('en_core_web_sm')
# load Berkeley Neural Parser https://github.com/nikitakit/self-attentive-parser 
# https://www.aclweb.org/anthology/P18-1249 
# we use the benepar_en2 pre-trained model for a less resource (memory and run-time) intensive architecture 
# for our system. the parser is integrated with spaCy, and the benepar_en2 is under 100mb
parser = benepar.Parser("benepar_en2")
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS #312 stopwords

# initialize a container for the tokens that should be removed (instead of the stopwords)
to_remove = ["\t", "      "] #for get_book_word_tokens

######## __for get_concreteness__ #########

concrete_df = pd.read_excel("./supportdata/ConcretenessRatings/Concreteness_ratings_Brysbaert_et_al_BRM.xlsx")
# there is a word "NaN" that is read as null value in pandas. causes problems when searching and slicing
# convert all values in "Word" column to str to solve this
concrete_df["Word"] = concrete_df["Word"].astype(str)

# there are "words" in the corpus that are actually bi-grams. e.g. baking soda. let's take leave them out 
concrete_df = concrete_df[concrete_df["Bigram"]==0]

# we want to calculate an average concreteness score for the entire corpus. let's try to recentre the scores 
# around 0. i.e. from -2.5 to 2.5. Would be good to review the paper to understand the motivations behind the
# chosen 5-point score. 
concrete_df["Conc.M"]=concrete_df["Conc.M"].apply(lambda x: x-2.5)

######## __processing functions__ #########

def process_one_author(author_id, functions, sent_per_inst=1, readpath = "./data/booksample_txt/", 
                       write_path="./processeddata/data_rawtxt/", overwrite_files = False):
    '''A wrapper fucntion that processes all the books for one author stored in corpus
    Inputs: author_id - the number of the author in a corpus;
            functions - an iterable of function(s) 
            readpath - path to folder that has the .txt files containing the author's texts
            write_path - path to folder to write the processed data to
            overwrite_files - boolean; to wipe existing files before starting to write to them
    Outputs: a collection (a dict) of processed information from one author's texts accepted into the corpus 
    Result: write, instance by instance, into different .txt files the results of subprocesses called.
    '''    
    # 0. save each set of data generated with different 
    write_path = write_path+str(sent_per_inst)+"_sent/"
    
    # 1. merge all the sentences from the author's books
    _ = []
    for filename in glob.iglob(readpath + author_id + "_" + "*.txt"):
        with open(filename) as file:
            _.extend(file.readlines())
    _ = "\t".join(_) 
    all_sents = _.split("\t")
    # 1a. depending on user input for sent_per_inst, there may be remainder sentences, cut all_sents here for cleaner 
    # execution in #5 further down
    slice_all_sents = int(len(all_sents)/sent_per_inst)*sent_per_inst # index to slice to 
                                                                      # == total num of sents eventually
    all_sents=all_sents[0:slice_all_sents]
    
    # 2. create directory if it's not there
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    
    # 3a. if user passes overwrite_files = True into function. open the files of the author 
    # with 'w' and close them (to wipe them)
    file_exts = ["_sentences.txt","_tokens.txt","_lemmas.txt","_pos.txt","_poswordpairs.txt",
                 "_parsetags.txt","_ners.txt","_concreteness.txt"]
    if overwrite_files == True:
        for file_ext in file_exts:
            try:
                open(write_path+author_id+file_ext, "w").close()
            except: 
                pass 
        
    #3b. open/create the files and write to it. we are writing lines, 
    # so we use "a+" to append instead of "w" or its variants
    sentencesfile = open(write_path+author_id+"_sentences.txt", "a+")
    tokensfile = open(write_path+author_id+"_tokens.txt", "a+")
    lemmasfile = open(write_path+author_id+"_lemmas.txt", "a+")
    posfile = open(write_path+author_id+"_pos.txt", "a+")
    poswordpairsfile = open(write_path+author_id+"_poswordpairs.txt", "a+")
    parsetagsfile = open(write_path+author_id+"_parsetags.txt", "a+")
    nersfile = open(write_path+author_id+"_ners.txt", "a+")
    concretenessfile = open(write_path+author_id+"_concreteness.txt", "a+")
    # a filelist to help with closing all the files at the end with a dict comprehension
    filelist = {"sentences":sentencesfile, "lemmas":lemmasfile,
                "tokens":tokensfile, "postags":posfile, "parsetags":parsetagsfile, 
                "poswordpairs":poswordpairsfile, "namedentities":nersfile, "concreteness":concretenessfile}
    
    # 4. a "global" dictionary storing feature information about every sentences. the keys of the 
    # dictionary are the features. 
    results_dict = {"sentences": list(), "lemmas": list(), "tokens": list(), "postags": list(), 
                    "poswordpairs": list(), "parsetags": list(), "namedentities": list(), "concreteness": list(),}
    
    # 5. go through every sentence of the author's 
    inst_counter = slice_all_sents/sent_per_inst # number of times we need to recursively process through all_sents 
    _start, _end = 0, sent_per_inst
    while inst_counter >0:  # while loop to implement the processing in clusters of sent_per_inst times
        instance = ".".join(all_sents[_start:_end]) # join all sentences in the cluster, with "." to handle cases
                                                    # of sentences that don't end with punctuation. 

        # create a spaCy document for the instance (i.e. sent_per_inst number of sentences )
        spacysentdoc = create_spacysentdoc(instance)
        _function_name = None

        for function in functions: 
        # initialize a list to store the output other than NE
            _result = []
            # initialize a storage for NE
            _entities = {"places": list(), "persons": list(), "dates": list()}

            # apply a function passed as a parameter
            # for NE extractions, append the output to the lists in the dictionary
            # as keys will remain the same
            if function == get_namedentities:
                _function_name, _function_results = get_namedentities(spacysentdoc)

                # write to file
                nersfile.write("\t".join(_function_results["places"])+"\t\t")
                nersfile.write("\t".join(_function_results["persons"])+"\t\t")
                nersfile.write("\t".join(_function_results["dates"])+"\t\t")
                nersfile.write("\n")          
                # extend lists in dict
                _entities["places"].extend(_function_results["places"])
                _entities["persons"].extend(_function_results["persons"])
                _entities["dates"].extend(_function_results["dates"])

            # for constituency parsing, we don't pass tags into the function
            elif function == get_parsetags: 
                # support functions return tuples. a result name (for easy file and dict_key 
                # selection here)
                _function_name, _function_results = function(spacysentdoc)
                # write to file
                filelist[_function_name].write("\t".join(_function_results)+"\t\t")
                filelist[_function_name].write("\n")
                # add to local results container
                _result.extend(_function_results)

            # for get_poswordpairs, special treatment because the results are a set of tuples
            elif function == get_poswordpairs: 
                _function_name, _function_results = function(spacysentdoc)
                # a special concat of the tuples for writing to file
                _function_results_str = ["()".join(ele) for ele in _function_results]
                filelist[_function_name].write("\t".join(_function_results_str)+"\t\t")
                filelist[_function_name].write("\n")
                _result.extend(_function_results)

            # for all other functions
            else:
                _function_name, _function_results = function(spacysentdoc)
                filelist[_function_name].write("\t".join(_function_results)+"\t\t")
                filelist[_function_name].write("\n")
                _result.extend(_function_results)

            if _function_name != None:
                # add to global container 
                if _function_name == "namedentities": 
                    results_dict[_function_name].append(_entities)
                else: 
                    results_dict[_function_name].append(_result)
       
        _start += sent_per_inst #update the _start, _end index with the size of sent_per_inst
        _end += sent_per_inst
        inst_counter -= 1  # decrement the inst_counter
        
    # close all the files with list_comp]
    [filelist[file].close() for file in filelist]
    
    # return global container. this contains all the features for every of the author's sentence
    return results_dict

def generate_dataframe(authornum, author_dict, select_postags=None, select_parsetags=None):
    '''
    A dictionary containing the features generated from the process_one_author process
    Inputs: authornum - str, the unique authornum for the author
            author_dict - dict, the output for one author from process_one_author
            select_postags - list, of Universal Dependencies parts-of-speech tags, if none passed, sets to default
            select_parsetags - list, of Stanford constituency parse tags, if none passed, sets to default
    Outputs: a pandas dataframe 
    Result: writes 
    '''
    
    # the structure of the values are lists, except the one for namedentities which has keys for 
    # places, persons and dates. 
    # 1. create a dataframe that appends for each pos tag 
    # 2. create an initial dataframe with n rows, the first column is the authornum, as well as literary movements  
    if select_postags == None: 
        # all UD POS tags 
        select_postags = ["ADJ","ADP","ADV","AUX","CONJ","CCONJ","DET","INTJ","NOUN","NUM",
                         "PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X","SPACE"]
    if select_parsetags == None: 
        select_parsetags = ["S", "NP", "VP"]
    
    author_numsents = len(author_dict['sentences'])
    # list of author_numsents length, filled with the authornum. for adding to the df at the end (prediction labels) 
    authornum_col = [authornum]*author_numsents
    
    all_sent = []
    for sentence_num in range(author_numsents):   
        one_sent = {}
        # the dict (and subsequent nested dict/list) structure 
        # is intended for easy and fast conversion to pd.DataFrame at the end
        for key in author_dict:
            if key== "postags":
                pos_counter = {i:[0] for i in  select_postags}
                for postag in author_dict[key][sentence_num]: 
                    try: 
                        pos_counter[postag][0]+=1
                    except: 
                        pass
                for key2 in pos_counter:
                    one_sent["pos_"+key2] = pos_counter[key2][0]                
            
            elif key== "parsetags":
                parse_counter = {i:[0] for i in  select_parsetags}
                for parsetag in author_dict[key][sentence_num]: 
                    try: 
                        parse_counter[parsetag][0]+=1
                    except: 
                        pass
                for key2 in parse_counter:
                    one_sent["parse_"+key2] = parse_counter[key2][0]

            elif key == "namedentities":
                # under the namedentities key, there are 3 other dictionary keys
                col_names = [ne_type for ne_type in author_dict[key][sentence_num]]
                ne_dict = {ne_type:[] for ne_type in col_names}
                for col_name in col_names:
                    ne_dict[col_name] = author_dict[key][sentence_num][col_name]
                for key2 in ne_dict:
                    one_sent["ne_"+key2] = ne_dict[key2]

            elif key == "concreteness" or key == "sentences":
                col_names = key
                values = author_dict[key][sentence_num]
                one_sent[key] = values[0]
            
            else: 
                col_names = key 
                values = author_dict[key][sentence_num]
                one_sent[key] = values
             
        all_sent.append(one_sent)
    # place the sentences into a pandas dataframe
    all_sent_df = pd.DataFrame(all_sent)
    # add the authornum information as the first column of the df
    all_sent_df.insert(loc=0, column="authornum", value=authornum_col)
    # lambda expression to calculate the length of the sentence (number of tokens) 
    # in each row, add the result as a new column in the df. 
    all_sent_df["sent_length"] = all_sent_df["tokens"].apply(lambda x: len(x))
    return all_sent_df

def create_spacysentdoc(sentence):
    '''
    Creates a spaCy document object from a string.
    Inputs: a single string, making up a sentence 
    Outputs: an annotated spaCy document
    '''
    # remove lingering anomalous characters
    sentence = re.sub(r'([-]+)|([“_])', "", sentence)  
    spacysentdoc = nlp(sentence)
    return spacysentdoc

def get_sentence(spacysentdoc):
    '''
    Returns the sentence that was used to create a spaCy document object.
    Inputs: spacysentdoc - spaCy document
    Outputs: a spaCy document with annotated information
    '''
    return "sentences", [str(spacysentdoc)]

def get_tokens(spacysentdoc):
    '''
    Returns tokens in a file
    Inputs: spacysentdoc - spaCy document
    Outputs: list of strings - lowercased tokens; with removed punctuation
    and custom undesired tokens
    '''
    return "tokens", [token.text.lower() for token in spacysentdoc \
                      if not token.is_punct and token.text not in to_remove]

def get_lemmas(spacysentdoc, tag = None):
    '''
    Lemmatizes the tokens of a specific POS or all tokens
    Inputs: spaCy annotated document, optional - tag, a specified POS tag
    The default value None would process all the tokens
    Outputs:a tuple containing the function name and a list of lemmas for the selected tokens
    '''
    if tag == None:
        lemmas = [token.lemma_ for token in spacysentdoc \
                  if not token.is_punct and token.text not in to_remove]
    else:
        lemmas = [token.lemma_ for token in spacysentdoc \
                  if token.pos_== tag and not token.is_punct and token.text not in to_remove]
    return "lemmas", lemmas

def get_postags(spacysentdoc):
    '''
    Performs postagging on a file
    Inputs: spaCy annotated document
    Outputs: a tuple containing the function name and the result (a list of tuples with token)
    Uses WordNet postags: https://spacy.io/api/annotation#pos-tagging
    '''
    return "postags", [token.pos_ for token in spacysentdoc]

def get_poswordpairs(spacysentdoc):
    '''Performs postagging on a file
    Inputs: spaCy annotated document
    Outputs: a tuple containing the function name and the result (a list of tuples with token)
    Uses WordNet postags: https://spacy.io/api/annotation#pos-tagging
    '''
    return "poswordpairs", [(token.pos_, token.text) for token in spacysentdoc]

def get_parsetags(spacysentdoc):
    '''Parses a sentence using Stanford CoreNLP constituency parsing
    Inputs: a list of sentences
    Outputs: a tuple containing the function name and a list of parse trees for all the sentences in a list
    '''    
    # getting the sentence from the spacysentdoc, so as to align with 
    # all other function methods (to use the process_one_author wrapper function)
    parsetags=[]
    try:
        parsetree = parser.parse(get_sentence(spacysentdoc)[1][0])
        # find only the non-terminal nodes
        parsetags_raw = re.findall( r"\[*Tree\('[A-Z]+", parsetree.__repr__()) 
        # using the nltk.Tree.__repr__ to bypass the common gs installation issue
        # https://stackoverflow.com/questions/39007755/cant-find-ghostscript-in-nltk?noredirect=1&lq=1

        # extract only the non-terminal symbol 
        parsetags = [tag.split("('")[1] for tag in parsetags_raw]
        return "parsetags", parsetags
    except:
        return "parsetags", parsetags

def get_namedentities(spacysentdoc):
    ''' Extracts named entities of place, person, date types
    Inputs: spaCy annotated document
    Outputs: a tuple containing the function name and a dictionary 
    with keys = NE types, values = corresponding tokens
    '''
    namedentities = dict()
    places = [ent.text for ent in spacysentdoc.ents if ent.label_ == 'GPE']
    persons = [ent.text for ent in spacysentdoc.ents if ent.label_ == 'PERSON']
    dates = [ent.text for ent in spacysentdoc.ents if ent.label_ == 'DATE']
    namedentities["places"] = places
    namedentities["persons"] = persons
    namedentities["dates"] = dates
    return "namedentities", namedentities
        
def get_concreteness(spacysentdoc):
    '''given a spacysentdoc (tokens of a sentence), returns the concreteness
    score for the sentence. The score is the sum of the score for tokens that can be 
    found in the http://crr.ugent.be/archives/1330 dataset. 
    Inputs: spacysentdoc
    Outputs: a tuple containing the function name and the result. result is an integer rounded to 3 dp 
    and converted to a string
    ''' 
    c_score = 0
    for token in [token.text for token in spacysentdoc]: 
        try:
            c_score += concrete_df[concrete_df["Word"] == token]["Conc.M"].values[0]
        except:
            pass
    return "concreteness", [str(round(c_score,3))]

def get_authabstract(author_id, authorabstracts_df, language="en"):
    '''
    searches the dataframe generated from utils_loaddataframe, storing all of an author's information to 
    retrieve a specified author's wikipedia abstracts
    returns a tuple dictionary containing the author number and an author's wikipedia abstracts matching the 
    language value input into the function
    '''
    author_abstracts = authorabstracts_df.loc[author_id]["authorabstracts"]
    abstract = [authorabstracts_df[language] if authorabstracts_df[language] else [] for i in author_abstracts]
    
    return author_id, abstract 

