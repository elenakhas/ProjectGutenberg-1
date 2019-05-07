import glob
import utils_preprocessing
import utils_statsgenerator
import os
import re

readpath = "./data/booksample_txt/"


def process_one_document_pipeline(filename, readpath, writepath):
    '''A wrapper fucntion that processes each text file separately and writes the results into a series of files.
    Inputs: path to a file, name of the file, path to the output files
    Outputs: text files for one book containing the processing results:
    '''
    features = ["sentence", "tokens", "ner", "pos", "trees", "lemmas", "un_lemmas", "un_tokens"]
    doc = utils_preprocessing.create_a_doc([filename])
    sentences = utils_preprocessing.segment_sentences(doc)
    tokens = utils_preprocessing.get_book_word_tokens(doc)
    lemmas = utils_preprocessing.lemmatization(doc)
    for feature in features:
        filewrite = writepath + filename.lstrip(readpath).rstrip(".txt") + "." + feature + ".txt"
        with open(filewrite, 'w+', encoding="utf-8") as fr:
            if feature == "sentence":
                fr.write("\n".join([sent.text for sent in sentences]))

            if feature == "tokens":
                fr.write("\n".join(tokens))

            if feature == "ner":
                nes = utils_preprocessing.ne_extraction(doc)
                for k in (nes.keys()):
                    fr.write("%s: \n %s" % (k, "\t".join(nes[k])) + "\n\n")

            if feature == "pos":
                pos = utils_preprocessing.postagging(doc)
                for p in (pos.keys()):
                    if not p == "PUNCT":
                        fr.write("%s: \n %s" % (p, "\t".join(pos[p])) + "\n\n")

            if feature == "lemmas":
                fr.write("\n".join(lemmas))

            if feature == "un_lemmas":
                fr.write("\n".join(set(lemmas)))

            if feature == "un_tokens":
                fr.write("\n".join(set(tokens)))

            if feature == "trees":
                trees = utils_preprocessing.const_parsing(sentences, n=10)
                subtrees_np = utils_preprocessing.get_sub_trees(trees, tag="NP", n=10)                
                for subtree_np in subtrees_np:
                    fr.write("NPS: \n")
                    fr.write(str(subtree_np) + "\n")
                
                subtrees_vp = utils_preprocessing.get_sub_trees(trees, tag="VP", n=10)
                fr.write("VPs: \n")
                for subtree_vp in subtrees_vp:
                    fr.write(str(subtree_vp) + "\n")

            print("File %s" % filewrite + " is written")

        fr.close()
    return


def process_book_corpus(readpath, writepath):
    '''Reads all books in the corpus, processes each book and writes files containing processing of each book
    Inputs: a path to the file, a path to the output file'''
    try:
        os.mkdir(writepath)
    except:
        pass

    for filename in glob.iglob(readpath + "*.txt"):
        process_one_document_pipeline(filename, readpath, writepath)
    return "All files are written"


def process_abstracts(df, language="en"):
    '''
    Reads abstracts in the corpus, processes each abstract and writes corresponding files
    Inputs: a dataframe, specified language
    '''
    writepath = "./abstracts/"
    try:
        os.mkdir(writepath)
    except:
        pass
    for i in df.index:
        abstracts = df["authorabstracts"].at[i]
        author = df["authornum"].at[i]
        abstract = abstracts[language]
        features = ["sentence", "tokens", "ner"]
        doc = utils_preprocessing.string_to_doc(abstract)
        sentences = utils_preprocessing.segment_sentences(doc)
        tokens = utils_preprocessing.get_book_word_tokens(doc)
        for feature in features:
            filewrite = writepath + author + "." + feature + ".txt"

            with open(filewrite, 'w+', encoding="utf-8") as fr:

                if feature == "sentence":
                    fr.write("\n".join(sentences))

                if feature == "tokens":
                    fr.write("\n".join(tokens))

                if feature == "ner":
                    nes = utils_preprocessing.ne_extraction(doc)
                    for k in (nes.keys()):
                        fr.write("%s: \n %s" % (k, "\t".join(nes[k])) + "\n\n")

                print("File %s" % filewrite + " is written")
        fr.close()
    return

def fill_dataframe(df):
    '''Fills the dataframe with the numerical values - number of occurrences of several features:
    POS distribution, vocabulary size, certain types of named entities, min, max and average sentence length
    Inputs: a dataframe
    Outputs: a modified dataframe with inserted columns
    '''
    columns = ["voc_size", "avg_sent", "min_sent", "max_sent", "ne_places", "ne_persons", "ne_dates", "tok_num"]
    postags = ["NOUN", 'VERB', 'ADJ', "ADV", "AUX", "INTJ", "NUM", "PRON", "PROPN", "PUNCT", "OTHER"]

    # create empty columns in a data frame
    for entry in columns + postags:
        df[entry.lower()] = 0
    #for each author, perform the calculations and fill in the information into a dataframe
    for i in df.index:
        author = df.at[i, "authornum"]
        doc = utils_preprocessing.process_an_author(author)
        df["voc_size"].at[i] = len(set(utils_preprocessing.lemmatization(doc)))

        sentences = utils_preprocessing.segment_sentences(doc)
        sentence_counts = utils_statsgenerator.words_per_sentence(sentences)
        df["avg_sent"].at[i] = utils_statsgenerator.average_sent_length(sentence_counts)
        df["min_sent"].at[i] = utils_statsgenerator.minmax(sentence_counts)[0]
        df["max_sent"].at[i] = utils_statsgenerator.minmax(sentence_counts)[1]
        df["tok_num"].at[i] = len(utils_preprocessing.get_book_word_tokens(doc))

        nes = utils_preprocessing.ne_extraction(doc)
        df["ne_places"].at[i] = len(nes["places"])
        df["ne_persons"].at[i] = len(nes["persons"])
        df["ne_dates"].at[i] = len(nes["dates"])

        pos = utils_preprocessing.postagging(doc)
        for tag in postags:
            df[tag.lower()].at[i] = len(pos[tag])

    return df
