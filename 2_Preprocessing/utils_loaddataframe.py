import pandas as pd
import glob

def loaddataframe( datapath="./data"):

    # load the json mongo exports for the authors and books info into pandas dataframes
    authors_df = pd.read_json(datapath+"/mongo_dumps/jsondump_authors_mongo.json")
    books_df = pd.read_json(datapath+"/mongo_dumps/jsondump_books_mongo.json")

    # the books collection (imported into books_df) contains the eventual set of authors and books selected for the 
    # corpus we want to join both dataframes on the set of authornums in books_df. 

    # 1. make a copy of authors_df so we don't work on the original data
    corpus_authors_df = authors_df.copy()
    # 2. we set the index for copy of the authors_df to the values in "authornum"
    corpus_authors_df = corpus_authors_df.set_index(corpus_authors_df["authornum"])
    # 3. with the index set, we  can slice the dataframe to get only the authornums present in books_df
    corpus_authors_df = corpus_authors_df.loc[list(books_df["authornum"].copy().unique())]
    # 4. we also set the index for books df to authornums. we need this for the join below
    books_df = books_df.set_index(books_df["authornum"])

    # 5. use pd.concat for both dfs. use inner join. place the smaller df on the left. (we know all authornums in 
    # books_df are present in corpus_authors_df)
    corpus_authorbook_df = pd.concat([corpus_authors_df.copy(), books_df], axis=1, join="inner")

    # 7. get the book txt file names 
    filenames = glob.glob(datapath+"/booksample_txt/*")
    files=[]
    for filename in filenames:
        _ = filename.rstrip(".txt")
        _ = _.split("/")[-1]
        authornum = _.split("_")[0]
        booknum = _.split("_")[1]
        file_dict ={"authornum":authornum, "booknum":booknum,"filename":filename }
        files.append(file_dict)
    files_df = pd.DataFrame(files)

    files_df.set_index(files_df["booknum"])

    # 8. since both dfs are of the same height use pd.merge for both dfs. 
    # use inner join by default. but before that, corpus_author_df has booknum in
    # int64, files_df has booknum as strings (from the split from the filename)
    files_df["booknum"]=files_df["booknum"].astype("int64")
    corpus_authorbook_df = pd.merge(corpus_authorbook_df.copy(),files_df, on="booknum")


    # 9. add the book titles to the dataframe
    titles = []
    for row in corpus_authorbook_df.index:
        for booknum in corpus_authorbook_df["books_info"][row]: 
            if str(corpus_authorbook_df.loc[row,"booknum"]) == booknum:
                titles.append(corpus_authorbook_df.loc[row,"books_info"][booknum])
    corpus_authorbook_df.loc[:,"booktitle"] = titles

    # 10 drop the books_info and other columns that are not necessary
    corpus_authorbook_df.drop(columns=["books_info", "_id", "authornum_x"], inplace=True)
    corpus_authorbook_df.rename(columns={'authornum_y':'authornum'}, inplace = True)
    
    return corpus_authorbook_df
