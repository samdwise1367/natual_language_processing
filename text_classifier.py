#text classification
import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)),category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15)) #get the most 15 used words in the document

#print(all_words['stupid'])

#set limit for number of words
word_features = list(all_words.keys())[:3000]

def features(doc):
    words = set(doc)
    feat = {}
    for w in word_features:
        feat[w] = (w in words)

    return feat

print(features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(features(rev),category) for (rev, category) in documents]

