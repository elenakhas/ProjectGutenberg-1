# ProjectGutenberg Corpus Builder and Machine Learning Tasks
This is a submission for a Natural Language Processing project towards the fulfilment of the requirments for the Data Science course conducted in Spring 2019 at Institut des Sciences du Digital, Management et Cognition, Université de Lorraine. 

__Team members: __

* _Balard, Srilakshmi_
* _Han, Kelvin_
* _Khasanova, Elena_ 


### Part Two: Processing text, extracting and visualising descriptive statistics
In this part of the project, we began the pre-processing for the corpus collected, as well as conducting some exploratory data analysis and visualisations. 

The preprocessing involved loading each of the .txt files containing sentences from authors' books and passing them through a pre-processing pipeline in order to extract intermediate features (tokens, lemmas, parts of speech tags, constituency parsing tags), as well as named entity recognition. 

Our pipeline is designed to allow flexibility in specifying the dataset structure. In particular, we enable the choice of processing inputs on varying levels (as sentences, paragraphs and documents) without the need to modify our code base. It is also possible to collect multiple (sentences, paragraphs and documents) and process them as one single instance/datapoint. 

The preprocessed data for a single author is exported out line by line into text files, and subsequently loaded into a pandas dataframe and pickled. Additionally, we include the loading, searching and computing of a concreteness score, taking  [data](http://crr.ugent.be/archives/1330) collected by a researchers from Ghent University, Belgium and McMaster University, Canada. Their dataset contains responses collected from subjects in a psycholinguistics setting. It measures the abstract notion of the "concreteness" connotated by a word, and contains ratings for more than 40,000 English lemmas. 

Additionally, we partially processed each author's English wikipedia abstracts. We focused on extracting named entities (particularly dates) and cleaning them.  For instance, we extract all 4-digit numerical information, which typically denotes an author's year of birth and death, or the period of being active in publication. Our further data cleaning included rounding the extract years to the nearest decade, so as to avoid downstream data sparsity issues. The extracted information could be useful for downstream training/prediction tasks, although we did not use the data for our subsequent classification and clustering tasks.  


__Technicalities:__ 

Our code was developed on [Jupyter](https://jupyter.org/) notebooks. For collobarative working hosted on [Google Colaboratory](https://colab.research.google.com/), which facilitated quick and dynamic pre-production experimentation. We also leveraged the following Python packages in our program: 

| package   	| usage  	|
|---	|---	|
| [spaCy](https://spacy.io/) 	| for our custom preprocessing pipepine covering tokenisation, lemmatisation, named entity recognition, constituency parsing 	|
| [benepar](https://github.com/nikitakit/self-attentive-parser)  	| a neural constituency parser that works with spaCy (and NLTK)	|
| [pandas](https://pandas.pydata.org/)  	| for creating dataframes for easy data handling, inspection and analysis   	|
| [numpy](https://www.numpy.org/)  	| for working with arrays  	|
| [matplotlib](https://matplotlib.org/)  	| for visualisations	|
| [wordcloud](https://github.com/amueller/word_cloud)  	| for setting up word cloud visualisations	|

as well as these system libraries for convenience functions: re, glob, time, os, pickle, string, collections

Architecture choices: 
1. We selected spaCy for our pipeline because of its flexibility to work with different pre-trained models as well as community developed tools. It integrates major frameworks currently populate within the NLP research and enterprise communities, such as Universal Dependencies tags as well as interface with task-specific software such as Rasa NLU (for conversational agents), AllenNLP (for a broad suite of NLP tools), and the Berkeley Neural Parser. 
2. We chose to use the Berkeley Neural Parser developed by UC Berkeley's NLP department (and presented with state-of-the-art accuracy results in mid-2018 [see ACL 2018 proceedings](https://www.aclweb.org/anthology/P18-1249)) and. Compared to the typically used Stanford Parser that requires the local or hosted deployment of a CoreNLP server (requiring large - more than 1gb - .jar files), the Berkeley parser has a significantly smaller footprint (an English model of about 100mb), as well as include models for [10 other languages](https://github.com/nikitakit/self-attentive-parser#available-models), enabling multilingual parsing. Further the parser can be easily installed through PyPi and its model can be downloaded and loaded with spaCy.

__Instructions for use:__

We provide our Project Gutenberg corpus tool in a package with the following: 

* a. a ['ProjectGutenberg_ProcessingEDA.ipynb'](https://github.com/hankelvin/ProjectGutenberg/blob/master/2_Preprocessing/ProjectGutenberg_ProcessingEDA.ipynb) Jupyter notebook with code, comments and details of our approach in preprocessing the data and conducting exploratory data analysis on the intermediate data generated.
* b. a ['utils_tokeniser.py'](https://github.com/hankelvin/ProjectGutenberg/blob/master/2_Preprocessing/utils_tokeniser.py) file for import to support the preprocessing in ProjectGutenberg_ProcessingEDA._
* c. a ['utils_loaddataframe.py'](https://github.com/hankelvin/ProjectGutenberg/blob/master/2_Preprocessing/utils_loaddataframe.py) file for import to support the generation and export of pandas DataFrames for authors in ProjectGutenberg_ProcessingEDA. _
* d. a ['utils_statsgenerator.py'](https://github.com/hankelvin/ProjectGutenberg/blob/master/1_DataExtraction/utils_statsgenerator.py) file for import to support the generation of visualisations in the EDA in ProjectGutenberg_ProcessingEDA. _
* d. a ['pg_dataextraction.py'](https://github.com/hankelvin/ProjectGutenberg/blob/master/2_Preprocessing/pg_dataextraction.py) file for loading objects defined in the previous corpus collection phase. _
* e. a ['PG-eng-author-min3v2019424.pickle'](https://github.com/hankelvin/ProjectGutenberg/blob/master/2_Preprocessing/PG-eng-author-min3v2019424.pickle) file that contains a pickled version of a completely populated GutenbergCorpusBuilder object together with associated Author instances for authors (and their books) admitted into our corpus. _
* f. a ['data'](https://github.com/hankelvin/ProjectGutenberg/tree/master/2_Preprocessing/data) folder containing exports of the corpus in various formats. 
    * /booksample_txt: Each plain text file contains k sentences from each author's oeuvre. k is set at the level based on the following parameters: sent_num/min_books. This generates a corpus that is evenly spread, in terms of number of books and number of sentences, across all authors admitted into the corpora. 
    * /data/mongo_dumps: Two json files. One json file contains the entire collection of authors information exported from a mongoDB database instance, the other contains the entire collection of books information exported from a mongoDB database instance. These are used to load author-book-sentences information to retrieve and extract data to generate the intermediate features. 
* g. ['processeddata'](https://github.com/hankelvin/ProjectGutenberg/tree/master/2_Preprocessing/processeddata) folder that contains these sub-folders: 
    * /data_rawtxt: containing txt. for each specific intermediate featureset for each author. 
    * /df_pickle_abstracts: containing pickle files that hold pandas dataframes generated for all author (processed information from their Wikipedia English abstracts)
    * /df_pickle_movements: containing pickle files that hold pandas dataframes generated for each author (with information about their literary movements extracted from DBPedia)
    * /df_pickles: containing pickle files that hold pandas dataframes generated for each author (with sentences, pos information, constituency parses etc)
* h. ['supportdata'](https://github.com/hankelvin/ProjectGutenberg/tree/master/2_Preprocessing/supportdata) folder that contain the files for other datasets used (and can potentially be used) for the feature generation. 
   
- - - - - - - - - - - - - - - 

1. To understand our approach, we recommend starting with the ProjectGutenberg_ProcessingEDA.ipynb. It contains all the relevant code as well as detailed step-by-step comments and analysis for our program and the preprocessing. Additionally, we have taken care to provide clear docstrings and code comments throughout our three util python scripts above.  
3. To use/examine our intermediate features, the best approach is to load pickled pandas dataframe files in ['processeddata'](https://github.com/hankelvin/ProjectGutenberg/tree/master/2_Preprocessing/processeddata)_ 
