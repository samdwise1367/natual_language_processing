#chunking is the process by which we group various words together bu their part of speech tags
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_data = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_data)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_pcontent():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            chunked.draw()

    except Exception as e:
        print(e)

process_pcontent()