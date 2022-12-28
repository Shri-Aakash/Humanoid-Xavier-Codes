import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bagOfWords(tokenizedSentence,words):
    tokenizedSentence=[stem(w) for w in tokenizedSentence]
    bag=np.zeros(len(words),dtype=np.float32)
    for idx,w in enumerate(words):
        if w in tokenizedSentence:
            bag[idx]=1
    return bag
