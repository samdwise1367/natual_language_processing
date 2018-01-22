#Stemming takes word and give the root stem of the word. Example runing will give run
# I was talking a ride in the car.
#I was riding in the car.
#porter stemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

ex_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in ex_words:
    print(ps.stem(w))

new_text = "It is very important to be python;y while you are pyhtoning with python. All pythoners have pythoned"
words = word_tokenize(new_text)
for t in words:
    print(ps.stem(t))