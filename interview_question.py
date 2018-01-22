#I made use of python3 instead of python 2
#I installed gensim and ntlk using pip install
#pip install genism
#pip install ntlk
#Author: Samson Oni
import logging
import time
from gensim import corpora,similarities,models
import pandas as pd
from nltk.corpus import stopwords

def vectorize(sentence, corpus):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) #log
    stopWords = set(stopwords.words('english'))  # generate stop words using NLTK Module
    filtered_corpus = [[w for w in words.lower().split() if w not in stopWords] for words in corpus] #remove stop words from the sentenses in the corpus

    corpus_dictionary = corpora.Dictionary(filtered_corpus)
    corpus_count = [corpus_dictionary.doc2bow(i) for i in filtered_corpus]

    sentence = corpus_dictionary.doc2bow(sentence.lower().split())
    corpus_dictionary.save('/tmp/corpus_dictionary.dict') #save the dictionary for future reference

    return sentence, corpus_count

def compare_distances(input_sen_vec, corpus_vecs, corpus):
	# TODO: YOUR ANSWER HERE!
    # transformation
    tfidf = models.TfidfModel(corpus_vecs)  # using a gensim model
    # transforming vector
    corpus_tfidf = tfidf[corpus_vecs]
    corpus_dictionary = corpora.Dictionary.load('/tmp/corpus_dictionary.dict') #load dictionary saved
    lsi_model = models.LsiModel(corpus_tfidf, id2word=corpus_dictionary, num_topics=2)  # initialize an LSI transformation
    vector_lsi = lsi_model[input_sen_vec]
    # initializing query
    index = similarities.MatrixSimilarity(lsi_model[corpus_vecs])

    # performing query
    sims = index[vector_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    index_result = sims[0][0] #to get the word_id of the most similar query
    most_similar_sentence = corpus[int(index_result)] #retrieve the exact sentence from the database list using the word_id

    return most_similar_sentence


def find_similar_sentence(input_sentence, corpus_file):
    start_time = time.time()

    readfile = pd.read_table('LEAVES OF GRASS') #used pandas to read the file
    corpus = readfile['LEAVES OF GRASS'].tolist() #convert pandas dataframe to list

    input_sen_vec, corpus_vecs = vectorize(input_sentence, corpus)
    most_similar_sentence = compare_distances(input_sen_vec, corpus_vecs, corpus)
    print('Took ', time.time() - start_time, 'seconds!')
    return most_similar_sentence

most_sim_sentence = find_similar_sentence('I sing the electric body', 'LEAVES OF GRASS')
print('most_sim_sentence: ', most_sim_sentence)

