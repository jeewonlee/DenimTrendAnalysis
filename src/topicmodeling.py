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
import pickle

#Topic Modeling
def topic_modeling(df, topics):
    vectorizer, X, features = vec(df)
    run_kmean(vectorizer, X, features, topics)
    run_nmf(vectorizer, X, features, topics)
    run_PCA(vectorizer, X, features, topics)

def vec(df):
    stop_words = get_stop_words('en')
    stop_words.extend(['nigga','niggas''sexiaws','qigfa2', 'http','saturdayonline','nigga','wear', 'denim','today','tomorrow','dick','saturdaynightonline', 'p9', 'romeo', 'playlyitjbyp9romeo','romeoplaylyitj','night', 'day', 'yesterday', 'wearing','tonight','every','pair'])
    #All tweets
    #Topic0
    stop_words.extend(['lol','people','never','work','fuck','im','hot','take'])
    #topic1
    stop_words.extend(['order','welcome','reseller','promo'])
    #topic2
    stop_words.extend(['keep','can','eyes','meet','ur','closer','holdin'])
    #topic3
    stop_words.extend(['play','please','hey','thank', 'thanks','playp9romeo','wewantp9onsno'])
    #topic4 - none
    #topic5
    stop_words.extend(['hate','much','really','decide','fucking','shopping','fucking','absolutely'])
    #topic6
    stop_words.extend(['don','floor','lying'])
    #topic7
    stop_words.extend(['room','anymore','tell','none'])
    #topic8
    stop_words.extend(['new','need','got','buy','get','want','buying','finally'])
    #topic9
    stop_words.extend(['find'])
    #topic10
    stop_words.extend(['pills','hands','everything','now','pulled','apart'])
    #topic11
    stop_words.extend(['feel','feels','shit','right','quick','good'])
    #topic12
    stop_words.extend(['finding','none','don','perfectly','anymore'])
    #topic13
    stop_words.extend(['want','school','dropped','got'])
    #topci14
    stop_words.extend(['got','shes','wants','everyone','everybody','saturday'])
    #topic15
    stop_words.extend(['need','pair','pairs'])
    #topic16
    stop_words.extend(['don','know','even','anymore','people','understand','care','get','want','think'])
    #topic17
    stop_words.extend(['can','wait','belive','work','decide'])
    #topic18
    stop_words.extend(['come','dat','yo'])
    #topic19
    stop_words.extend(['got','know','exactly','just','finally','ve'])
    #topic20
    stop_words.extend(['need'])
    #topic21
    stop_words.extend(['need','buy','get','tee'])
    #topic22
    stop_words.extend(['trying', 'need','tryna','tell','mean','work','get'])
    #topic23
    stop_words.extend(['just','bought','found'])
    #topic24
    stop_words.extend(['followed','world'])
    #topic25
    stop_words.extend(['put','told','step','jump','gotta','struggle','trying'])
    #topic26
    stop_words.extend(['way','playp9romeo','queronotvz','hear','hey'])
    #topic27
    stop_words.extend(['like','pairs'])
    #topic28
    stop_words.extend(['walked','eyes','burn','made','know', 'like'])
    #topci29
    stop_words.extend(['need','buy','shopping','asap','really','go'])
    #topic30
    stop_words.extend(['first','time','wore','remember','haven','year','month'])
    #topic31
    stop_words.extend(['still','squeeze','perfect','much'])
    #topci32-none
    #topic33-none
    #topic34-none
    #topic35
    stop_words.extend(['teenage','dream','hands','put','let','racing','heart','imma'])
    #topic36
    stop_words.extend(['look','good','make','nice'])
    #topcic37,38,39-none
    #topci40
    stop_words.extend(['hug','see','thru'])
    #topic41
    stop_words.extend(['size','now','heart','defined','went','judged'])
    #topic42
    stop_words.extend(['really','uncomfortable','made','realize','want','like'])
    #topic43
    stop_words.extend(['find','can','hard','never','will'])
    #topic44
    stop_words.extend(['hearing','good','little','tea','kisses','hearin','likes'])
    #topic45
    stop_words.extend(['bought','just','rock','said'])
    #topic46
    stop_words.extend(['friday','cold','beer','radio','fried','cheicken','just'])
    #topic47
    stop_words.extend(['store'])
    #topic48
    stop_words.extend(['got','necessary','can'])
    #topic49
    stop_words.extend(['eyes','meet','holding','closer','won','ever','meet','close','closer','til','eyes'])

    #v1 stopwords
    #topic0
    stop_words.extend(['ll','best','money','life','always'])
    #topic1
    stop_words.extend(['alone','till','home','inside','photograph'])
    #topic2
    stop_words.extend(['love','pleaseeee','um','girl'])
    #topic3,4-none
    #topic5
    stop_words.extend(['gene'])
    #topic7
    stop_words.extend(['kinda','well','kinda','ass','making','stare'])
    #topic8-none
    #topic12
    stop_words.extend(['insde'])
    #topic14
    stop_words.extend(['labor'])
    #topic15
    stop_words.extend(['wish','comfy','confortable'])
    #topic16
    stop_words.extend(['life'])
    #topic17
    stop_words.extend(['weather'])
    #topic18
    stop_words.extend(['dad','pretty','sure','women','math','hell'])
    #topic19
    stop_words.extend(['whatever','mood','ring'])
    #topic20
    stop_words.extend(['looking','left'])
    #topic 21
    stop_words.extend(['favorite'])
    #topic 22
    stop_words.extend(['plz','barely','qusetion','cause','actually'])
    #topic24
    stop_words.extend(['didn','try','em','wanna','wears'])
    #topic25
    stop_words.extend(['something','follow','foot','five','x5'])
    #topic26
    stop_words.extend(['gonna'])
    #topic27
    stop_words.extend(['worn','since','months','worn','years','forever','week','havent','worn','shouldn','should','things','looks'])
    #topic28,31
    stop_words.extend(['one','question','thing','direction'])
    #topic33
    stop_words.extend(['lamo','might','ima','yeah'])
    #topic 36
    stop_words.extend(['kid','help','undo','ll','extended','swag','song','another','miss'])
    #topic 37
    stop_words.extend(['getting','big','small'])
    #topic 38
    stop_words.extend(['enough','hair','taking','com','fall'])
    #topic 39
    stop_words.extend(['whole','club','fall'])
    #topic 40
    stop_words.extend(['wanna','kinda','wanna'])
    #topic 42
    stop_words.extend(['damn','shake','one','ya','wears'])
    #topic 43
    stop_words.extend(['actually','might','comfortable','wow','combined','talking'])
    #topic 44
    stop_words.extend(['last','honestly','since','cant','dont','years','couldn'])
    #topic 45
    stop_words.extend(['show'])
    #topic 46
    stop_words.extend(['changin','putting','life','three','dryer','lotion','two','four'])
    #topic 47
    stop_words.extend(['putthing','feeling'])
    # stop_words.extend(['im','never','one','keep','meet','eyes', 'play','room','anymore','new','buy','bought','want','got','buying','finally','everything','feel','feels','can','none','anymore','finding'])
    # stop_words.extend(['school','dropped','lying','floor','band','everybody','saturday','need','pair','don','know','even','understand'])
    # stop_words.extend(['can','never','decide','believe','get','got','come','look','like','tell','mean','lol','followed','gotta','hear','love','new','looks','pairs','go','made','remember','still'])
    # stop_words.extend(['much', 'fuck','imma','bought','really','something','follow','see', 'thats','show','thru','defined','judged','uncomfortable','realize','hate','find','can','trying','looks'])
    # stop_words.extend(['good','hearing','wears','wear','said','told','right','meet','friday','beer','radio','chicken','fried','now','rihgt','now','necessary','wanna','ever','alone','won','til','holdin','keep','holding'])
    # stop_words.extend(['hands','now','good','hand','hands','see','let','put','get','quick','apart','seams','pulled','none','anymore','care','tryna','just','playp9romeo','queronotvz','thank'])
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['text'].fillna(''))
    # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
    # X = vectorizer.fit_transform(df['text'].fillna(''))
    features = vectorizer.get_feature_names()
    return vectorizer, X, features

def print_top_words(model, feature_names, n_top_words):
    allwords = []
    with open('../analysis/analysis_v1.txt','ab') as f:
        f.write("\n\nAll Data\n")
        f.write(model.__class__.__name__)
        f.write("\n")
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            words=(",".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print words
            allwords.append(words)
            f.write("Topic #%d:" % topic_idx)
            f.write(words)
            f.write("\n")
        print()
    f.close()
    return allwords

# def print_top_words2(model, feature_names, n_top_words):
#     allwords = []
#     for topic_idx, topic in enumerate(model.components_):
#         print("Topic #%d:" % topic_idx)
#         words=(" ".join([feature_names[i]
#                         for i in topic.argsort()[:-n_top_words - 1:-1]]))
#         print words
#         allwords.append(words)
#     print()
#     return allwords

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
    with open('../../picklefiles/nmf_v1.pkl', 'wb') as output:
        mat = nmf.fit_transform(X)
        pickle.dump(mat, output, pickle.HIGHEST_PROTOCOL)
    print("\nFirst stopwords edit, Topics in NMF model:")
    #tfidf_feature_names = vectorizer.get_feature_names()
    print_top_words(nmf, features, 20)
    return mat

def run_SVD(vectorizer, X, features, topics):
    svd = TruncatedSVD(n_components=topics)
    with open('../../picklefiles/svd_v1.pkl', 'wb') as output:
        mat = svd.fit_transform(X)
        pickle.dump(mat, output, pickle.HIGHEST_PROTOCOL)
    print("\nFirst stopwords edit, Topics in truncated SVD model:")
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
    with open('../analysis/analysis_v1.txt','ab') as f:
        for i in np.arange(20):
            print i, ": ", vals[i]
            temp = i+': '+vals[i]
            f.write(str(temp))
    f.close()

# def save_model(model):
#     with open('models.pkl', 'wb') as output:
#         pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    #print 20, "topics"
#    print "2013 first quarter"
    #df = pd.read_csv('../../cleandata/2013Q1.csv', delimiter=';')
    #df = pd.read_csv('../../data/cleandata/2013Q1_temp.csv', delimiter=';')
    df = pd.read_csv('../../cleandata/all.csv', delimiter=';')
    #df = df[100:]
    vectorizer, X, features = vec(df)
    #run_kmean(vectorizer, X, features, 20)
    nmf_mat = run_nmf(vectorizer, X, features, 50)
    svd, n_topics, svd_mat = run_SVD(vectorizer, X, features, 50)
    svd_val(n_topics, svd)
    #lda_mat = run_LDA(vectorizer, X, features, 50)
#    print "nmf mat", nmf_mat.shape()
#    print "svd mat", svd_mat.shape()
#    print "lda mat", lda_mat.shape()
