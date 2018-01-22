import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_data = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenzer = PunktSentenceTokenizer(train_data)
tokenized = custom_sent_tokenzer.tokenize(sample_text)

def processing():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            named = nltk.ne_chunk(tagged, binary=True)

            named.draw()

    except Exception as e:
        print(e)
processing()