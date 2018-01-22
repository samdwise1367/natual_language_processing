#with wordnet we can look up words and the meaning
from nltk.corpus import wordnet

syno = wordnet.synsets("program")

#synset
print(syno[0].name())

#just the word
print(syno[0].lemmas()[0].name())

#definition
print(syno[0].definition())

#examples
print(syno[0].examples())

#get synonyms and antonyms
synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())


print(set(synonyms))
print(set(antonyms))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cactus.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))