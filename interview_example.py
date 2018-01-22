import logging
import time
from collections import defaultdict
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora,similarities,models
import pandas as pd
from nltk.corpus import stopwords
start_time = time.time()

readfile = pd.read_table('LEAVES OF GRASS')
corpora_to_list = readfile['LEAVES OF GRASS'].tolist()

stopWords = set(stopwords.words('english'))


#remove common words
filtered_corpus = [[w for w in words.lower().split() if w not in stopWords]for words in corpora_to_list]
# frequency = defaultdict(int)
# for text in filtered_corpus:
#     for token in text:
#         frequency[token] += 1
#
# filtered_corpus = [[token for token in text if frequency[token] > 1]
#           for text in filtered_corpus]
#set dictionary
print(filtered_corpus)
corpus_dictionary = corpora.Dictionary(filtered_corpus)

print(corpus_dictionary)
#save the dictionary
corpus_dictionary.save('/tmp/corpus_dictionary.dict')

corpus_count = [corpus_dictionary.doc2bow(i) for i in filtered_corpus]
corpora.MmCorpus.serialize('/tmp/corpus_count.mm',corpus_count)

# print(corpus_count)
# print(corpus_dictionary)
# print(filtered_corpus)

#transformation
tfidf = models.TfidfModel(corpus_count) # using a gensim model

#transforming vector
corpus_tfidf = tfidf[corpus_count]
# for i in corpus_tfidf:
#     print(i)
lsi = models.LsiModel(corpus_tfidf, id2word=corpus_dictionary, num_topics=2) # initialize an LSI transformation

#doc = "I sing the electric body"
doc = "To the Garden the World"
vec_bow = corpus_dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]
# print(vec_lsi)

#initializing query
index = similarities.MatrixSimilarity(lsi[corpus_count])

#performing query
sims = index[vec_lsi]
#print(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])
# print(sims) # print sorted (document number, similarity score) 2-tuples
# print('Took ', time.time() - start_time, 'seconds!')

# index_result = sims[0][0]
# print(int(index_result))
#
# #get sentence from intial list
# similar_result = corpora_to_list[int(index_result)]
# print(similar_result)