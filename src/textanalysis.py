import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import NMF, LatentDirichletAllocation

def topic_modeling(filename):
    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

    # 1. Apply k-means clustering to the twitter
    df=pd.read_csv(filename, delimiter=';')
#    vectorizer = TfidfVectorizer(stop_words='english')
#    X = vectorizer.fit_transform(df['text'].fillna(''))
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'].fillna(''))
    features = vectorizer.get_feature_names()
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X)

    # 2. Print out the centroids.
    print filename
    print "cluster centers:"
    #print kmeans.cluster_centers_

    # 3. Find the top 10 features for each cluster.
    top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
    print "top features for each cluster:"
    for num, centroid in enumerate(top_centroids):
        print "%d: %s" % (num, ", ".join(features[i] for i in centroid))

    topics =10
    #4.NMF
    nmf = NMF(n_components=topics, random_state=1,alpha=.1, l1_ratio=.5).fit(X)
    print("\nTopics in NMF model:")
    tfidf_feature_names = vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, topics)

    #5. LDA python algorithm
    lda = LatentDirichletAllocation(n_topics=topics,max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit_transform(X)

    print("\nTopics in LDA model:")
    tf_feature_names = vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, topics)


if __name__ == '__main__':
    topic_modeling('../../data/twitter/201301.csv')
    topic_modeling('../../data/twitter/201302.csv')
    # topic_modeling('../../data/twitter/201303.csv')
    # topic_modeling('../../data/twitter/201304.csv')
