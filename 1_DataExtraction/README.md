# ProjectGutenberg Corpus Builder and Machine Learning Tasks

This is a submission for a Natural Language Processing project towards the fulfilment of the requirments for the Data Science course conducted in Spring 2019 at Institut des Sciences du Digital, Management et Cognition, Université de Lorraine. 

__Team members: __


* _Balard, Srilakshmi_
* _Han, Kelvin_
* _Khasanova, Elena_ 

### Part One: Extracting and storing RDF information and text

In this part of the project, we extracted information from three sources of information: (i) the [Project Gutenberg website](https://pages.github.com/), containing information about more than 58,000 copyright-free books and their authors; (ii) [Wikipedia](www.wikipedia.org), the largest multi-lingual user-contributed source of information in the world; and (iii) the [DBpedia project](www.dbpedia.org) which contains mahcine-generated relational informatioon from Wikipedia data. 

From (i), we obtained a list of authors that meet certain publishing requirements (language, minimum number of books, roles in his/her books' publication) as well the machine-readable texts of the books. From (ii), we obtained summary information on authors in multiple languages. From (iii) we obtained information on the literary movements each author is associated with. We note that for (ii) and (iii), not all authors on Project Gutenberg have wikipedia pages and amongst those that do have Wikipedia (and DBpedia) pages, many of their DBpedia entries do not have any literary movement labels. 

__Corpus information:__ 

We extracted sentences for 109 authors that have between 3 and 30 English language books that they had written themselves (i.e. not as contributors, editors or translators etc). From these 109 authors we drew 250 sentences from at least 3 of their books randomly. The sentences are drawn with the goal of ensuring a consistent spread in terms of number of books (and sentences per book) for each author represented in the corpus. A small number of authors, particularly those whose books are exceptionally short, saw sentences draw from more than 3 books (between 4 and 5). We don't believe this will significantly alter or impact the performance of any model built on the data. 
    
We took significant care to ensure that no Project Gutenberg metadata (present in all the files for books on the website) was selected for entry into the corpus. In addition, we took measures to try and exclude generic publishing data (e.g. copyright information, content pages etc). 

__Technicalities:__ 

Our code was developed on [Jupyter](https://jupyter.org/) notebooks. For collobarative working hosted on [Google Colaboratory](https://colab.research.google.com/), which facilitated quick and dynamic pre-production experimentation. We also leveraged the following Python packages in our program: 

| package   	| usage  	|
|---	|---	|
| requests and [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) 	| for retrieving and parsing the "Browse by Author" pages on Project Gutenberg 	|
| [SPARQLWrapper](https://rdflib.github.io/sparqlwrapper/)  	| for passing SPARQL queries in Python onto a DBpedia SPARQL endpoint   	|
| [wikipedia](https://github.com/goldsmith/Wikipedia)  	| for extracting author wikipedia pages in a structured manner   	|
| [nltk](https://www.nltk.org/)  	| for tokenising sentences from each book  	|
| [pymongo](https://api.mongodb.com/python/current/)  	| for setting up a mongoDB database, to store the corpus	|

as well as these system libraries for convenience functions: csv, datetime, time, random, collections, string, re, urllib, os, pickle. 

__Instructions for use:__

We provide our Project Gutenberg corpus tool in a package with the following: 

_* a. a ['pg_dataextraction.py'](https://github.com/hankelvin/ProjectGutenberg/blob/master/1_DataExtraction/pg_dataextraction.py) file for execution. _
* b. a ['ProjectGutenberg_DataExtraction.ipynb'](https://github.com/hankelvin/ProjectGutenberg/blob/master/1_DataExtraction/ProjectGutenberg_DataExtraction.ipynb) Jupyter notebook with details and explanations of our approach as well as general analysis and code for the creation of a local copy of a [mongoDB](https://www.mongodb.com/) database. 
* c. a ['randomstate.pickle'](https://github.com/hankelvin/ProjectGutenberg/blob/master/1_DataExtraction/randomstate.pickle) file that contains a pickled version of the seed state we used to generate our corpus. This will allow any other user to replicate the generatuion of our corpora on another machine. 
* d. a ['PG-eng-author-min3v2019419.pickle'](https://github.com/hankelvin/ProjectGutenberg/blob/master/1_DataExtraction/PG-eng-author-min3v2019420.pickle) file that contains a pickled version of a completely populated GutenbergCorpusBuilder object together with associated Author instances for authors (and their books) admitted into our corpus. _
* e. a ['data'](https://github.com/hankelvin/ProjectGutenberg/tree/master/1_DataExtraction/data) folder containing exports of the corpus in various formats. 
    * /booksample_txt: Each plain text file contains k sentences from each author's oeuvre. k is set at the level based on the following parameters: sent_num/min_books. This generates a corpus that is evenly spread, in terms of number of books and number of sentences, across all authors admitted into the corpora. 
    * /booksample_csv: Each file has the same content as those in /booksample_txt. 
    * /wholebook_csv: Each file contains the entire text for an admitted book into the corpus, except Project Gutenberg boilerplate metadata as well as a percentage of the text file removed as a buffer (to exclude general publisher information). 
    * /data/mongo_dumps: Two json files. One json file contains the entire collection of authors information exported from a mongoDB database instance, the other contains the entire collection of books information exported from a mongoDB database instance.
   
- - - - - - - - - - - - - - - 

1. To understand our approach, we recommend starting with the ProjectGutenberg_DataExtraction.ipynb. It contains all the relevant code as well as detailed step-by-step comments and analysis for our program and the corpus generated. 
2. To replicate the collection of our corpus (or collect corpora, by generating a new random seed and/or changing the parameter settings), use the pg_dataextraction.py file. 
3. To use our corpus, the best approach is to load and ingest the .txt plaintext files in /data/booksample_txt in your machine learning scripts or Jupyter notebooks _ 

__Overview of our approach__

Our solution is supported via two classes, their attributes and associated functions. The first class - GutenbergCorpusBuilder - is intended to handle author-title mining on the Project Gutenberg website, filtering and selection as well as storage of overall corpus data. The attributes of this class are designed for easy ingestion into a mongoDB, or similar non-relational database. The other class - Author - is intended to hold the processes for accessing text files for books, processing them and storing them. The Author class also contains methods for collecting 

To build a corpus, a user will only need to interact with 2 methods from the GutenbergCorpusBuilder. These are namely, in the order of intended use: (i) get_library; and (ii) populate_corpus. 

The first method - __get_library__ - will crawl all of the 'Browse by Author' pages on the Project Gutenberg website and collect author information (including books he/she authored as well as wikipedia pages). By passing the 'min_book', 'max_book' as well as 'languages' and 'roles' parameters, the user can balance the content of the corpus in terms of author and book numbers, as well as have it filtered based on language(s) and author role(s). We note in particular, that the author role setting could be become significant for certain machine learning tasks (for instance incorporating books where an author is merely an editor or contributor could lead to degraded model performance for an author-genre classification task). 

The default values for this function, as well as our setting for the corpus generated are as follows: 

|Parameter	|Default setting	|Setting for this corpus	|
|---	|---	|---	|
|min_books   	|1   	|3   	|
|max_books   	|float(inf)   	|30   	|
|languages   	|'all'   	|['english']   	|
|roles   	|'all'   	|['as author']   	|


The parameters, their types and default settings are designed with the intention to allow the collection of all books available on Project Gutenberg. For the purpose of our collected corpus, we have chosen the parameters so as to obtain a balanced corpus that selects major as well as minor authors. 

The second method - __populate_corpus__ - will take the pre-filtered list of author and their books, instantiate an Author object, and begin collecting carefully data on the author and his/her books. At the background, the function will first retrieve all of the literary movements the author is associated with, from DBpedia. The primary information bottleneck lies with these literary movement labels - not all authors have DBpedia pages and for those that do, many do not have literary movements associated to them. As such, we only proceed with the next steps of adding an author to the final corpus if he/she has these literary movement labels. For authors that pass this filter, the method proceeds to extract and store the multilingual abstracts for the author. Finally, it proceeds to randomly pick a set of the author's book's (the size of this set is the same for all authors and equivalent to the min_books set for the corpus), and clean and segment the text files in a list of sentences. The cleaning is intended to exclude boilerplate Project Gutenberg metadata as well as book publisher information. An option is provided to clean the text files by excluding the Project Gutenberg metadata more precisely, albeit this extends the time required to process and collect the corpus.  

The default values for this function, as well as our setting for the corpus generated are as follows: 

|Parameter	|Default setting	|Setting for this corpus	|
|---	|---	|---	|
|sent_num   	|250   	|250   	|
|precise_clean   	|False   	|True   	|

