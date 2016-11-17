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
import jw_tokenize as tw
from langdetect import detect

#create dataframe
def create_df(filename1, filename2, filename3):
    df = pd.read_csv(filename1,delimiter=';')
    df.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5']
    df1 = pd.read_csv(filename2,delimiter=';')
    df1.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5']
    df2 = pd.read_csv(filename3,delimiter=';')
    df2.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5']
    df=df.append(df1, ignore_index=True)
    df=df.append(df2, ignore_index=True)
    return df

#remove badwords
def remove_badword(df):
    bad_index=[]
    for index, tweet in enumerate(df.text):
        if 'niggas' in str(tweet).decode("utf8").lower():
            bad_index.append(index)
    return df.drop(df.index[bad_index]), bad_index

#Remove tweets without word jeans
def remove_noise(df):
    noise_index = []
    for index, tweet in enumerate(df.text):
        if 'jeans' not in str(tweet).decode("utf8").lower():
            noise_index.append(index)
    return df.drop(df.index[noise_index]), noise_index

#Remove advertisement
def filtering_add(df):
    ad_index = []
    for index, tweet in enumerate(df.text):
        url = tw.Url_RE.search(str(tweet))
        if url:
            ad_index.append(index)
    return df.drop(df.index[ad_index]), ad_index

#Remove other tweet with other language
def filter_nonen(df):
    nonen_index = []
    lan_list=[]
    for index, tweet in enumerate(df.text):
        lan = detect(str(tweet).decode("utf8"))
        print index
        if lan !='en':
            nonen_index.append(index)
            lan_list.append(lan)
    return df.drop(df.index[nonen_index]), nonen_index, lan_list

#Topic Modeling
def topic_modeling(df, topics):
    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

    # 1. Apply k-means clustering to the twitter
    # topics = 2
    #df=pd.read_csv(filename, delimiter=';')
    vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,3))
    X = vectorizer.fit_transform(df['text'].fillna(''))
    # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
    # X = vectorizer.fit_transform(df['text'].fillna(''))
    features = vectorizer.get_feature_names()
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
        print "%d: %s" % (num, ", ".join(features[i] for i in centroid))


    #4.NMF
    nmf = NMF(n_components=topics, random_state=1,alpha=.1, l1_ratio=.5).fit(X)
    print("\nTopics in NMF model:")
    tfidf_feature_names = vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, 20)

    #5. LDA python algorithm
    # lda = LatentDirichletAllocation(n_topics=topics,max_iter=5,
    #                                 learning_method='online',
    #                                 learning_offset=50.,
    #                                 random_state=0)
    # lda.fit_transform(X)
    #
    # print("\nTopics in LDA model:")
    # tf_feature_names = vectorizer.get_feature_names()
    # print_top_words(lda, tf_feature_names, topics)

if __name__ == '__main__':
#     df = create_df('../../data/201403.csv','../../data/201404.csv','../../data/201405.csv')
#     #df.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5']
#     noise_df, noise_arr =remove_noise(df)
#     ads_df, ads_index = filtering_add(noise_df)
#     nonen_df, nonen_index, lan_lst = filter_nonen(ads_df)
#     final, bad_index =remove_badword(nonen_df)
#     final.to_csv('../../cleandata/2014Q2.csv', sep=';')
      for i in np.arange(5,10):
          print i, "topics"
          print "2013 second quarter"
          df = pd.read_csv('../../cleandata/2013Q2.csv', delimiter=';')
          topic_modeling(df, i)
          print "2014 second quarter"
          df1 = pd.read_csv('../../cleandata/2014Q2.csv', delimiter=';')
          topic_modeling(df1, i)
          print "2015 second quarter"
          df2 = pd.read_csv('../../cleandata/2015Q2.csv', delimiter=';')
          topic_modeling(df2, i)
