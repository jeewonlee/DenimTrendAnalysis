import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import jw_tokenize as tw
from langdetect import detect
from stop_words import get_stop_words

#Topic Modeling
def topic_modeling(df, topics):
    vectorizer, X, features = vec(df)
    run_kmean(vectorizer, X, features, topics)
    run_nmf(vectorizer, X, features, topics)
    run_PCA(vectorizer, X, features, topics)

def vec(df):
    stop_words = get_stop_words('en')
    stop_words.extend(['sexiaws','qigfa2', 'http','saturdayonline','nigga','wear', 'denim','today','tomorrow','dick','saturdaynightonline', 'p9', 'romeo', 'playlyitjbyp9romeo','romeoplaylyitj','night', 'day', 'yesterday', 'wearing','tonight','every','pair'])

    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['text'].fillna(''))
    # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
    # X = vectorizer.fit_transform(df['text'].fillna(''))
    features = vectorizer.get_feature_names()
    return vectorizer, X, features

def print_top_words(model, feature_names, n_top_words):
    allwords = []
    with open('../analysis/analysis.txt','ab') as f:
        f.write("2013Q1\n")
        f.write(model.__name__)
        f.write("\n")
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            words=(",".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print words
            allwords.append(words)
            f.write("Topic #%d:" % topic_idx)
            f.write(words)
        print()
    f.close()
    return allwords

def print_top_words2(model, feature_names, n_top_words):
    allwords = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        words=(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print words
        allwords.append(words)
    print()
    return allwords

def run_kmean(vectorizer, X, features, topics):
    # 1. Apply k-means clustering to the twitter
    # topics = 2
    #df=pd.read_csv(filename, delimiter=';')
    kmeans = KMeans(n_clusters=topics)
    kmeans.fit(X)

    # 2. Print out the centroids.
    #print filename
    print "cluster centers:"
    #print kmeans.cluster_centers_

    # 3. Find the top 10 features for each cluster.
    top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-21:-1]
    print "top features for each cluster:"
    for num, centroid in enumerate(top_centroids):
        " ".join(features[i] for i in centroid)
        print "%d: %s" % (num, words)

def run_nmf(vectorizer, X, features, topics):
    #4.NMF
    nmf = NMF(n_components=topics, random_state=1,alpha=.1, l1_ratio=.5)
    mat = nmf.fit_transform(X)
    print("\nTopics in NMF model:")
    #tfidf_feature_names = vectorizer.get_feature_names()
    print_top_words(nmf, features, 20)
    return mat

def run_SVD(vectorizer, X, features, topics):
    svd = TruncatedSVD(n_components=topics)
    mat = svd.fit_transform(X)
    print("\nTopics in truncated SVD model:")
    print_top_words(svd, features, 20)
    return svd, topics, mat

def run_LDA(vectorizer, X, features, topics):
    #5. LDA python algorithm
    lda = LatentDirichletAllocation(n_topics=topics,max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    mat=lda.fit_transform(X)

    print("\nTopics in LDA model:")
    # tf_feature_names = vectorizer.get_feature_names()
    print_top_words(lda, features, 20)
    return mat

def svd_val(n_topics, svd):
    vals = svd.explained_variance_ratio_
    for i in np.arange(20):
        print i, ": ", vals[i]

if __name__ == '__main__':
    #print 20, "topics"
    print "2013 first quarter"
    df = pd.read_csv('../../cleandata/2013Q1.csv', delimiter=';')
    #df = pd.read_csv('../../data/cleandata/2013Q1_temp.csv', delimiter=';')
    #df = pd.read_csv('../../data/data/201402.csv', delimiter=';')
    vectorizer, X, features = vec(df)
    #run_kmean(vectorizer, X, features, 20)
    nmf_mat = run_nmf(vectorizer, X, features, 50)
    svd, n_topics, svd_mat = run_SVD(vectorizer, X, features, 50)
    svd_val(n_topics, svd)
    #lda_mat = run_LDA(vectorizer, X, features, 50)
#    print "nmf mat", nmf_mat.shape()
#    print "svd mat", svd_mat.shape()
#    print "lda mat", lda_mat.shape()
