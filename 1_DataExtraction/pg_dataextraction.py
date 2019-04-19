import requests
from bs4 import BeautifulSoup
import wikipedia
from SPARQLWrapper import SPARQLWrapper, JSON
import csv, datetime, time, random, string, re, urllib, os, collections  
from nltk.tokenize import sent_tokenize
from urllib.error import HTTPError
import pickle 
with open('randomstate.pickle', 'rb') as f:
    random.setstate(pickle.load(f)) 
# We use random sampling in some of the functions of our program. 
# To ensure we can replicate the same dataset, we include the use
# of the same seed state whenever running this corpus builder.

class GutenbergCorpusBuilder: 
    '''
    initiates a GutenbergCorpusBuilder object which stores information about selected authors that are found on 
    the Project Gutenberg(PG) website.
    Authors are stored based on their unique PG numerical code. For each author, selected books and their 
    respective PG URL are stored.
    
    Inputs: corpusname - string representing name of the corpus being created.
    '''
    
    def __init__(self, corpusname):
        self.corpusname = corpusname
        self.corpusversion = "v"+ str(datetime.datetime.now().year) + str(datetime.datetime.now().month) +        str(datetime.datetime.now().day)
        
        # a dictionary containing dictionaries.
        # The top level keys - unique numbers for authors on the PG website,
        # the values -  dictionaries containing author information: 
        # keys - 'authorname', 'books_info'; wiki_info'; 
        # values - strings or embedded dictionaries:
        # authorname: string with extracted name;
        # books_info: dictionary: key - book ID, value - book title;
        # wiki_info: dictionary: key - language; value - wiki link extracted from PG
        #
        # {author ID: {authorname:'name', books_info:{bookID: book title}, wiki_info: {language: wiki link}}}
        self.authors = dict()
       
        # a dictionary containing sets of sentences selected from each author's filtered books;
        # the top level keys are the unique numbers for authors, the values are sets containing
        # sentences from an author's book (as strings).
        # {author ID: Author()}
        self.corpus = dict()
        
        self.min_books = int # the min_books value passed into the get_library method.
        self.max_books = int # the max_books value passed into the get_library method.
        
    def populate_corpus(self, sent_num=250, precise_clean=False):
        '''
        for each author in self.authors, generates an Author() class instance, populates all 
        attributes of the Author() class, adds to self.corpus.
        
        Inputs | sent_num: int - the total number of sentences to collect for a single author selected for the corpus. 
        '''
        if len(self.authors) > 0:
            _counter = 0
            for authornum in self.authors: # authornum is a key (an author's unique number)
                if authornum not in self.corpus.keys():
                    # instantiate an Author()
                    authorname = self.authors[authornum]["authorname"]
                    authorwiki_info = self.authors[authornum]["wiki_info"].copy() 
                    authorbooks_info_keys = list(self.authors[authornum]["books_info"].copy().keys())
                    
                    _author = Author(authorname=authorname, authornum=authornum, 
                                    min_books=self.min_books)
                    
                    # run the populate_attributes() to extract and process the information for the author
                    _author.populate_attributes(authorwiki_info=authorwiki_info, 
                                                authorbooks_info_keys=authorbooks_info_keys,
                                                sent_num=sent_num, precise_clean=precise_clean)
                    
                    # store to Author() to corpus 
                    self.corpus[authornum] = _author

                    # append wiki abstract info and literary movement to author's dictionary in 
                    # self.author so as to easily transmit each author's basic information into mongodb
                    self.authors[authornum]["authorabstracts"] = _author.authorabstracts
                    self.authors[authornum]["literarymovements"] = _author.literarymovements
                _counter += 1
                if _counter%100 == 0:
                    print("{} authors have been processed, out of {} authors in selections".format(_counter, len(self.authors)))
        else: 
            print("The authors attribute is empty, please run get_library first or             check the parameters passed into get_library.")
        
    
    def get_library(self, min_books=1, max_books=float("inf"), languages = "all", roles = "all"):
        '''
        Goes through the PG website's 'sort by author' pages. Extracts author and corresponding book 
        information that meet a number of selection criterion (see inputs). 
        
        Inputs | 
        1. min_books: int - the minimum number of books available for an author, which meets the languages 
        and roles parameters. Default value is 1. 
        5. languages: either a str "all", or a list containing the languages (in lowercase) to count towards 
        the author's min_books level. The list of languages available can be found here 
        https://www.gutenberg.org/catalog/. Default is "all". 
        6. roles: either a str "all", or a list containing the roles that an author can have in a book. 
        These include: Commentator, Translator, Contributor, Photographer, Illustrator, Editor.
        Default value is "all".
        Outputs | saves the results to self.authors
        '''
        charlist = []
        charlist[:0] =  [letter for letter in string.ascii_lowercase] + ["other"]

        library = dict()
        for char in charlist:
            # Team comment: we select the authors and books via the "Browse by Author" lists instead of  
            # the "Browse by Books" list. Although the latter has a more predictable page structure 
            # (i.e. 1 book name, followed by 1 author name, recursively), the former includes 
            # information about the Author's role in the book. We believe that this could have
            # a meaningful impact on the predictive capabilities for models on different tasks, 
            # especially at larger scale.
            
            link = 'https://www.gutenberg.org/browse/authors/'+ char
            page = requests.get(link)
            soup = BeautifulSoup(page.content, 'html.parser')
            one_letter = self._unite_authors_nums_books(self._get_authors_numsnames(soup)[0],                                                            self._get_authors_numsnames(soup)[1],                                                            self._get_bookswiki_info(soup)[0],                                                            self._get_bookswiki_info(soup)[1],                                                            min_books, max_books, languages, roles)
            
            library.update(one_letter)
            print("{} authors from the '{}' alphabetical category have been added.".format(len(one_letter),char))
            
            # del variable to clear memory
            del soup
            
            # Put the function to sleep for a randomised number of seconds (non-integer number between 
            # 0.5 and 4) to mimic human surfing patterns.
            time.sleep(random.uniform(0.5,4))
            
        self.authors = library
        self.min_books = min_books
        self.max_books = max_books
    
    def _get_authors_numsnames(self, soup):
        '''
        A helper function for _unite_authors_nums_books. Extracts all author names from a BeautifulSoup 
        copy  of a 'Browse by Author' page on the PG website. 
        
        Inputs | soup:a BeautifulSoup object - containg a copy of the PG 'Browse by Author' page. 
        Outputs | a tuple containing two lists: The first contains author's numbers on the page, the 
        second contains corresponding author's names on the page. 
        '''
        authornames = []
        # the author names are stored within the "name" attribute under each "a" class
        # use regex wildcard so that find_all will catch and return all "a names" with values
        authorname_BSlist = soup.find_all('a', {"name":re.compile("\w*")})

        for authorname in authorname_BSlist:
            # \- and \? to escape special characters. .rstrip to remove trailing whitespaces. 
            authornames.append(re.sub(r'[0-9,\-\?]*', '', authorname.text).rstrip())

        authornums = []
        # the author numbers are stored within the "href" attribute. Every line for a book 
        # on the page has a "title" attribute with the value "Link to this author". We will use
        # this to shift to only the lines with the author's number. 
        authornums_BSlist = soup.find_all('a', {"title":"Link to this author"})

        for authornum in authornums_BSlist:
            authornums.append(authornum["href"].lstrip("#"))

        return authornums, authornames

    def _get_bookswiki_info(self, soup):
        '''
        A helper function for _unite_authors_nums_books. Extracts all the book titles and numbers from a 
        BeautifulSoup copy of a 'Browse by Author' page on the PG website. Also extracts author wikipedia 
        link information if it is available on the PG website. 
        
        Inputs | soup:a BeautifulSoup object - containg a copy of the PG 'Browse by Author' page. 
        Outputs | a tuple containing two lists. 
          1. The first list contains dictionaries. Each dictionary contains information about an author's 
          books on PG. this includes: book titles, corresponding PG books numbers, the author's role in 
          each book, and the language of each book. 
          2. The second list contains dictionaries. Each dictionary contains information about an author's 
          wikipedia links on PG. An author's wiki dictionary may be empty, contain 1 link, or more than 1 
          link. 
        '''
        books_info = list()
        wiki_info = list()

        # content under the 'ul' tags: books, links as one list organized by ul
        authorsbooks_BSlist = soup.find_all('ul')
        # for each ul, access the content: books, links; each book is a bs object

        for author in authorsbooks_BSlist:
            # there are two classes of attributes within each ul tag. The book information
            # 1. title and book PG number is under the 'pgdbetext' class. 
            books_BSlist = author.find_all(class_='pgdbetext')

            authorbooks_info = {}
            for book in books_BSlist:
                # the book numbers are stored in the href attribute. e.g. "ebooks/19323"
                booknum = book.find('a')['href'].split("/")[-1]
                PG_booktitle = book.text

                # storing the information regarding a single author's books in a dictionary
                authorbooks_info[booknum]=PG_booktitle
            
            # appending the dictionary containing one author's books to a list
            books_info.append(authorbooks_info)
            
            # 2. for the author is/are under the 'pgdbxlink' class. 
            wiki_BSlist = author.find_all(class_='pgdbxlink')

            authorwiki_info = {}

            for wiki in wiki_BSlist:
                # 1. the wiki links are stored in the href attribute. 
                PG_wikilink = wiki.find('a')['href'] # get the whole link
                
                # some of the lines tagged "pgdbxlink" include "See also: xxx" links. 
                # we filter them out here
                if "wikipedia.org" in PG_wikilink:

                    # 2. because PG stores the link in URL-safe format (e.g. "\x" is "%"), we will face 
                    # issues with non-ASCII characters e.g. á whose URL-safe encoding cannot be passed 
                    # into the wikipedia package. use urllib.requests.unquote to resolve this 
                    # https://docs.python.org/2/library/urllib.html#utility-functions 
                    PG_wikilink = urllib.request.unquote(PG_wikilink)

                    # 3. get the language code for the wikipage
                    wikilang = re.findall(r'/\w+', PG_wikilink)[0].strip('/')
                    # storing the information regarding a single author's wikipedia links in a dictionary
                    authorwiki_info[wikilang] = PG_wikilink

            # appending the dictionary containing one author's wikipedia links to a list
            wiki_info.append(authorwiki_info)
            
        return books_info, wiki_info

    
    def _unite_authors_nums_books(self, authornums, authornames, books_info, wiki_info, min_books, 
                                  max_books, languages, roles):
        '''
        A helper function for get_library. 
        
        Inputs | 
        1. authornums:list - list of author numbers from a "sort by author" page on the PG website. 
        2. authornames:list - list of author names  from a "sort by author" page on the PG website. 
        3. books_info: list - a list containing dictionaries, each of which has information about 
        an author's books 
        4. wiki_info: list - a list containing dictionaries, each of which has information about 
        an author's wikipedia
        page, as provided by the PG website. There may be none, one, or more wikilinks for an author. 
        5. min_books:int - the minimum number of books available for an author, which meets the languages 
        and roles parameters. default value is 1 (since an author listed on PG will have at least 1 book 
        to his name).
        6, max_books:int - the minimum number of books available for an author, which meets the languages 
        and roles parameters. default value is infinity.
        7. languages:either a str "all", or a list containing the languages (in lowercase) to count towards the author's 
        min_books level. The list of languages available can be found here 
        https://www.gutenberg.org/catalog/. default is "all". 
        8. roles: either a str "all", or a list containing the roles (in lowercase) that an author can 
        have in a book. These include: commentator, translator, contributor, photographer, illustrator, 
        commentator, editor. default value is "all".
        Outputs | a dictionary containing PG numbers for authors who meet the min_books, languages and 
        roles requirements, as well as information each of these author's books. 
        '''
        # we want to be sure that the authornums, authornames, books_info, and wiki_info are aligned 
        # before proceeding to merge them. 
        try:
            assert len(authornums)==len(authornames) and len(authornums)==len(books_info) and len(authornums)==len(wiki_info)
        except AssertionError as e:
            e.args += ("The length of authornums, authornames and books_info do not match.",)
            raise
            
        authorbooks_info = dict()
        # if default parameters passed into the function, add all authors and their books to the corpus.  
        if min_books == None and languages == "all" and roles == "all":
            for i in range(len(authornums)):
                authorbooks_info[authornums[i]]=                        {"authorname": authornames[i], "books_info": books_info[i], "wiki_info": wiki_info[i]}
        else:
            # place languages and roles input in sets, for use in .intersection below. 
            languages_set = set(languages)
            roles_set = set(roles)
            
            for i in range(len(authornums)):
                author_bookset = books_info[i]
                _topop = []
                for book in author_bookset: 
                    
                    # using regex to find text in parentheses. Book language e.g. (English) and author role 
                    # e.g. (as Author) are contained in parentheses. Some books which are part of a series, 
                    # have (of N) in their titles too, where N is the number of books in that series. 
                    title_text_in_parentheses =                    re.findall(r'\(([a-zA-Z]+\s*[a-zA-Z]*[0-9]*)\)', author_bookset[book])
                    
                    # lowercase the text in parentheses and put it into sets. 
                    _title_text_in_parentheses =                    set([i.lower() for i in title_text_in_parentheses])
                    
                    # if languages is set to "all" or if the intersection of _title_text_in_parentheses
                    # and languages_set returns a non-empty set, pass to the next check. Otherwise add this 
                    # book number to the list of books to pop from this author_bookset
                    if languages == "all" or _title_text_in_parentheses.intersection(languages_set): pass
                    else:
                        _topop.append(book) 
                        continue 
                    # do the same for author's role as for language above
                    if roles == "all" or _title_text_in_parentheses.intersection(roles_set): pass
                    else:
                        _topop.append(book) 
                        continue    
                # pop the books that don't meet the language and role specifications. 
                for pop in _topop:
                    books_info[i].pop(pop)
                    
                # check if number of books meeting the language and role requirements meet the 
                # min_book requirement 
                if min_books <= len(books_info[i]) <= max_books:
                    authorbooks_info[authornums[i]]=                            {"authorname": authornames[i], "books_info": books_info[i], 
                             "wiki_info": wiki_info[i]}
                    
        return authorbooks_info 
    
    def __str__(self):
        return "There are {} authors entered in this corpus".format(len(self.corpus))


class Author:
    '''
    Initiates a Author object which stores information about a selected author that is available on 
    the Project Gutenberg(PG) website. Other information drawn from (i) DBPedia - author literary movements
    (ii) wikipedia - multilingual author abstract, (iii) PG - selected sentences from author's texts 
    
    Inputs: authorname: str, authornum:str, authorwiki_info: dict, authorbooks_info_keys:list of numbers in 
    string, min_books: int
    '''
    
    def __init__(self, authorname, authornum, min_books):
        '''
        initiates the Author object with the author's name. 
        
        Inputs | authorname: str, authornum:str, authorwiki_info: dict, 
        authorbooks_info_keys:list of author numbers (in str), min_books: int
        '''
        self.name = authorname
        self.number = authornum
        self.min_books = min_books  # the min_book setting at the GutenbergCorpus class that led
                                    # to this author's selection for the corpus
        
        # a dictionary with the book numbers as keys and lists as values. Lists  
        # contain strings that have been pre-processed by the segment_sentence method.
        self.processed_subcorpus = dict()       
        
        self.authorabstracts = dict() 
        self.literarymovements = list()
        
        
    def populate_attributes(self, authorwiki_info, authorbooks_info_keys, sent_num, precise_clean):
        '''
        A convenience function to call _build_subcorpus, _get_authorabstract and  _get_literarymovement, 
        which will respectively populate the processed_subcorpus, authorabstract and literarymovements
        attributes for this Author instance.  
        
        Inputs | authorwiki_info: dict, authorbooks_info_keys:list of numbers (in str), min_books: int
        Result | stores results to self.literarymovements, self.authorabstracts and self.processed_subcorpus 
        '''
        # check for /data directory, else create for storing files from _build_subcorpus
        if not os.path.isdir('./data'):  
            os.mkdir("data")
        
        self._get_literarymovement(authorwiki_info)
        
        # the information bottleneck is at the dbpedia literary movement labels.
        # multilingual wiki abstract and text processing requires a large amount of resources 
        # so we only do these for authors that we manage to get literary movements for.
        if len(self.literarymovements) > 0: 
            self._get_authorabstract(authorwiki_info)
            self._build_subcorpus(authorbooks_info_keys=authorbooks_info_keys, 
                                  sent_num=sent_num, precise_clean=precise_clean)
        
    
    def _build_subcorpus(self, authorbooks_info_keys, sent_num, precise_clean):
        '''
        A helper function for .populate_attributes. Selects the books of an author's to extract sentences
        from. the number of books is the same as min_book set for the corpus's author selection criteria.
        if an author has more books than min_books, a random sampling is done. a basic pre-processing to 
        remove PG metadata and publisher information is done next. results are written to two sets of csv files. 
        the first contains only selected sentences, the second contains the entire processed book. Additionally, 
        the selected sentences for each book are written to plaintext files. 
        
        Inputs | authorbooks_info_keys:list of numbers (in str), min_books: int
        Result | saves to two sets of csv files: (i) selected sentences only; (ii) entire pre-processed 
        book. selected sentences also saved to plaintext files. also stores (i) to self.processed_subcorpus
        '''
        
        _author_cleanbooks = dict()
        
        if len(authorbooks_info_keys) == self.min_books: 
            for booknum in authorbooks_info_keys: 
                all_sentencesinbook =                self._cleansegment_book(booknum=booknum, precise_clean=precise_clean)

                if len(all_sentencesinbook) > sent_num/self.min_books:
                    _author_cleanbooks[booknum] = all_sentencesinbook
        else: 
            # 1. recursively select a number of books until len(_author_cleanbooks) matches min_books
            #    at each recursion, apply _cleansegment_book on the book. If the cleaned book meets the 
            #    length requirement, add to _author_cleanbooks. Do this for up to 10 tries, failing
            #    which we will exclude the author and all of his/her books from the corpus.
            _tries = 0
            _unvisited = set(authorbooks_info_keys)
            while len(_author_cleanbooks) < self.min_books and _tries<=10 and len(_unvisited) > 0:
                # randomly select min_books number from author. if author only has min_books, 
                # sampling will return the same set
                try: # try to take a random sample (it could fail if _unvisited < sample size)
                    _newnums = set(random.sample(_unvisited, self.min_books-len(_author_cleanbooks)))
                    _unvisited = _unvisited.difference(_newnums)
                    for booknum in _newnums: 
                        all_sentencesinbook =                        self._cleansegment_book(booknum=booknum, precise_clean=precise_clean)

                        if len(all_sentencesinbook) > sent_num/self.min_books:
                            _author_cleanbooks[booknum] = all_sentencesinbook
                    _tries += 1
                except: # break the while loop if sampling fails
                    break 
                                
        # 2. if min_books still not met, move to return. this effectively excludes author from corpus 
        if len(_author_cleanbooks) < self.min_books:
            return 
        else:
            pass
        
        # 3. extract k number of sentences from each accepted author's books. k is the total number
        #   of sentences required for each author divided by the author min_books set for the corpus 
        _authors_sentences = dict()
        for booknum in _author_cleanbooks: 
            _sample =            random.sample(_author_cleanbooks[booknum], round(sent_num/self.min_books))
            # book to temporary dictionary, with the booknum as the key. 
            _authors_sentences[booknum] = _sample
        
        # 4. write the cleaned book and sampled sentences to file 
        for booknum in _authors_sentences: 
            self._write_tofile(booknum, all_sentencesinbook = _author_cleanbooks[booknum], 
                               sample_all_sentencesinbook = _authors_sentences[booknum])
        
        # 5. update the processed_subcorpus attribute for the author, with the sampled sentences
        self.processed_subcorpus.update(_authors_sentences)
        
        
    def _write_tofile(self, booknum, all_sentencesinbook, sample_all_sentencesinbook): 
        '''
        A helper function to export processed texts and lists of sentences to csv and plaintext files. 
        Called by _build_subcorpus. 
        '''
        # 1. write the entire cleaned and segmented book to a csv file. 
        if not os.path.isdir('./data/wholebook_csv'):
            os.mkdir("data/wholebook_csv")
        with open("./data/wholebook_csv/"+self.number+"_"+booknum+'.csv', 'a') as csv_file:
            # we set file open mode to 'a' to append to file instead of overwriting
            write_file = csv.writer(csv_file, dialect = 'excel')
            write_file.writerow(all_sentencesinbook)
            del csv_file # delete to free memory

        # 2a. write the book sample to a csv file
        if not os.path.isdir('./data/booksample_csv'):
            os.mkdir("data/booksample_csv")
        with open("./data/booksample_csv/"+self.number+"_"+booknum+'.csv', 'a') as csv_file:
            write_file = csv.writer(csv_file, dialect = 'excel')
            write_file.writerow(sample_all_sentencesinbook)
            del csv_file # delete to free memory

        # 2b. write the book sample to a txt file
        if not os.path.isdir('./data/booksample_txt'):
            os.mkdir("data/booksample_txt")
        with open("./data/booksample_txt/"+self.number+"_"+booknum+'.txt', 'a') as txt_file:
            txt_file.writelines("\t".join(sample_all_sentencesinbook))
            del txt_file # delete to free memory

            
    def _cleansegment_book(self, booknum, precise_clean, 
                           urlpath = "https://www.gutenberg.org/files/{}/{}.txt"):
        '''
        takes a booknum, navigates to the PG page with the .txt file for this book. uses urlopen to 
        retrieve the contents of this file. if precise_clean = False, only retrieves lines between the
        last "*START" and first "*END" line in the file. 
        
        Inputs | booknum: int - the unique number on PG for a book, urlpath: str - the url structure for a book's  
        page on PG, precise_clean: boolean 
        Outputs | all_sentencesinbook: list -  a list of sentences after the basic pre-processing 
        '''    
        book_content = []
        
        # open target_url with the urllib.request.urlopen() method,
        # for each line in response, decodes with the expected 
        # encoding format PG uses for plain .txt book files. 
        # see https://www.gutenberg.org/wiki/Gutenberg:Readers%27_FAQ#R.35._What_do_the_filenames_of_the_texts_mean.3F
        for extenc_pair in [('', 'ascii'), ('-0', "utf-8"), ('-8', 'ISO 8859-1')]: 
            # iterate through likely filename endings and associated encodings on PG
            try: 
                target_url = urlpath.format(booknum,booknum+extenc_pair[0])
                with urllib.request.urlopen(target_url) as response: 
                    for line in response: 
                        # urlopen reads as bytes, to ease processing, we decode to string.
                        # most PG .txt files are encoded in latin-1/ascii format. 
                        try:
                            book_content.append(line.decode(extenc_pair[1]))
                        except: # revert to latin-1 in the event of unexpected PG encoding behaviour 
                            book_content.append(line.decode("latin-1"))
                    response.close()
                    del response
            except HTTPError: 
                continue
                
        # remove PG metadata precisely, but slower to execute
        if precise_clean == True: 
            start_index = 0                # index for the start of the text
            stop_index = -1  # index for the end of the text  

            # Each PG book .txt file is ended with metadata marked with "* START" and "* END" or 
            # minor variations. * START-tagged metadata tend to, but don't always just, appear in 
            # the first 25% of the .txt file, and vice-versa for * END tagged metadata. we split 
            # the file in the top and bottom thirds and run searches for * START and * END (for 
            # some savings in search time)
            
            _2third_marker = round(len(book_content)*0.67)
                                         
            #1. search for *END tags from the back of the file, for two-thirds of the file
            for index_num in range(_2third_marker):
                if re.match(r'\*+\s*END ', book_content[-index_num]):
                    stop_index = -index_num
            
            #2. search for anomalous *START tags in the last two-thirds of the file, 
            #   but begining from the, possibly new, stop_index 
            for index_num in range(-stop_index, _2third_marker):
                # searching for the last * END from the back, in the last half of the file 
                if re.match(r'\*+\s*START ', book_content[-index_num]):
                    stop_index = -index_num
            
            #3. finally, search for the last START tag from the front, within the first two-thirds
            for index_num in range(_2third_marker):
                # searching for the last * START in the first half of the file 
                if re.match(r'\*+\s*START ', book_content[index_num]):
                    start_index = index_num 
            
            # slicing the section of the text between the start_index and stop_index. 
            book_content = book_content[start_index:stop_index]

        # join all the text without "\r\n" i.e. return carriage and newline 
        clean_book_content = " ".join([l.strip("\r\n") for l in book_content if l != "\r\n"])
        # use nltk's sent_tokenise
        all_sentencesinbook = sent_tokenize(clean_book_content)

        # strip first and last 10% of lines (as a buffer to avoid collecting generic publishing data)
        _10pc = round(len(all_sentencesinbook)*0.10)
        all_sentencesinbook = all_sentencesinbook[_10pc:-_10pc]

        return all_sentencesinbook


    def _get_authorabstract(self, author_wiki_info):
        '''
        A helper function for .populate_attributes. Gets available author abstract from wikipedia using
        the wikipedia python package. 
        
        Input | authorwiki_info: dict
        Result | stores results to self.authorabstracts
        '''
        _abstracts = {}

        for wikilang in author_wiki_info: 
        # set the language 
            wikipedia.set_lang(wikilang)
            wikiname = author_wiki_info[wikilang].split("/")[-1]

            try: # without disambiguation: we start with the presumption that PG has 
                # accurate author wikipedia links. set auto_suggest to False to prevent 
                # additional (unnecessary) handling of the author page name by the wikipedia package.
                wikipage = wikipedia.page(title=wikiname, auto_suggest=False)
                _abstracts[wikilang] = wikipage.summary

            except PageError: 
                print("There is a PageError resulting with this wikiname: {}".format(wiki_name) )
                pass 
            except DisambiguationError: 
                print("There is a DisambiguationError resulting with this wikiname: {}".format(wiki_name))
                pass 

        self.authorabstracts = _abstracts    
    
      
    def _get_literarymovement(self, authorwiki_info):
        '''
        A helper function for .populate_attributes. takes an author's name, makes a DBpedia query 
        with the name using the SPARQLWrapper package, 
        returns the literary movements that the author is associated with. 
        
        Input | authorwiki_info: dict
        Result | stores results to self.literarymovements
        '''
        if len(authorwiki_info) > 0:
            # since dbpedia is based off wikipedia, we will use the author's name as in 
            # the wikipedia link obtained from PG. 
            _authorwiki_info = authorwiki_info.copy().popitem()
            wikiname = _authorwiki_info[1].split('/')[-1].replace("_", " ")
            wikilang = _authorwiki_info[0]

            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            query = '''SELECT ?text
                WHERE {
                ?writer rdf:type dbo:Writer ;
                foaf:name %r @%s.
                {?writer dbo:genre ?genre .}
                UNION
                {?writer dbo:movement ?genre .}
                ?genre rdfs:label ?text
                FILTER (lang(?text) = "en")
                }''' %(wikiname, wikilang)
            # using %r for names to handle non-ascii wikinames that get passed as bytes in %s
            # see https://pyformat.info/ for e.g. "Bahá'u'lláh" becomes "Bahá\'u\'lláh"
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            genres = set()
            for i in range (len(results['results']['bindings'])):
                genre = results['results']['bindings'][i]['text']['value']
                genres.add((re.sub(r'\([^)]*\)', '', genre.lower())).rstrip())
            self.literarymovements = list(genres)


if __name__ == "__main__":
    min_books = 3
    max_books = 30
    sent_num = 250 
    precise_clean=True
    # instantiate a GutenbergCorpusBuilder 
    PGcorpus = GutenbergCorpusBuilder(corpusname="PG-eng-author-min{}".format(min_books))
    # start collecting and filtering author and book details from the Project Gutenberg site
    PGcorpus.get_library(min_books = min_books, max_books = max_books, 
                         languages = ["english"], roles = ["as author"])
    # read text files, select sentences, pre-process sentences, store to subcorpora
    PGcorpus.populate_corpus(sent_num=sent_num, precise_clean=precise_clean)




