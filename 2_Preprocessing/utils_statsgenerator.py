import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def _vocab_barplot(vocab_dict):
    '''
    an iterable list of datapoints 
    '''
    ig, ax = plt.subplots()
    plt.title("Author vocabulary size")
    for i in vocab_dict:
        ax.bar(i, vocab_dict[i])
    plt.show()    

def _sentsize_boxplot(data): 
    fig1, ax1 = plt.subplots()
    ax1.set_title('Author max, min, avg, sentence size')
    ax1.boxplot(data,vert=False, whis=0.75) 
    plt.show()

def _posdistributions(pos_dist, select_postags):
    '''
    '''
    fig, ax = plt.subplots()
    plt.title("Author POS distribution")
    for i in ["pos_"+postag.lower() for postag in select_postags]:
        ax.bar(i, pos_dist[i])
    plt.show()

def _makecloud(allsentences):
    '''
    '''
    plt.figure(figsize=[15,15])
    wordcloud = WordCloud().generate(allsentences)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# #1 .
# def words_per_sentence(sentences):
#     '''Returns a list containing lengths of each sentence
#     Inputs: sentences - a list of sentences
#     Outputs: a list of int - corresponding number of tokens in each sentence (with removed punct and "stop" tokens)
#     '''
#     lengths = []
#     for sentence in sentences:
#         sentence = nlp(sentence)
#         word_tokens = [token.text for token in sentence if not token.is_punct and not token.text in to_remove]
#         lengths.append(len(word_tokens))
#     return lengths

# def average_sent_length(sent_lengths):
#     '''Computes the average sentence length
#     Inputs: sent_lengths - a lst of sentence lengths
#     Outputs: a float value of an average sentence in a document
#     '''
#     return float("{0:.2f}".format(sum(sent_lengths)/len(sent_lengths)))