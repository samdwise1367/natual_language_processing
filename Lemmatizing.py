#Very similar to stemming. The end result will be a real word
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("Cats"))
