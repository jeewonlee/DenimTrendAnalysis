import os
import logging
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from stop_words import get_stop_words
#from gensim import copora
from collections import defaultdict
from pprint import pprint


if __name__ == '__main__':
    #print 'hey'
    df = pd.read_csv('../../cleandata/2015Q1.csv', delimiter=';')
    #df = pd.read_csv('../../201501.csv', delimiter=';')
    documents = df.text.tolist()
    stoplist = get_stop_words('en')
    texts = [[word for word in document.decode("utf8").lower().split() if word not in stoplist] for document in documents]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token]+=1

    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    #pprint(texts)
    model = Word2Vec(texts, size=100, window=5, min_count=5, workers=4)
    #model.save('../../picklefiles/word2vec_v1.txt')
    odel.most_similar(positive=['jeans'])
    print model.similarity('jeans','skinny')
