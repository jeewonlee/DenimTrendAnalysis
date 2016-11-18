import os
import numpy as np
import pandas as pd
#from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn.cluster import KMeans
from collections import Counter
#from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
#from sklearn.decomposition import NMF, LatentDirichletAllocation
#import jw_tokenize as tw
from langdetect import detect
#from stop_words import get_stop_words

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

#Remove tweets without word jeans
def remove_noise(df):
    noise_index = []
    for index, tweet in enumerate(df.text):
        if 'jeans' not in str(tweet).decode("utf8").lower():
            noise_index.append(index)
    return df.drop(df.index[noise_index]), noise_index

#Remove advertisement
def filtering_ads(df):
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

#remove badwords
def remove_badword(df):
    bad_index=[]
    for index, tweet in enumerate(df.text):
        if 'niggas' in str(tweet).decode("utf8").lower():
            bad_index.append(index)
    return df.drop(df.index[bad_index]), bad_index

if __name__ == '__main__':
    df = create_df('../../data/201212.csv','../../data/201301.csv','../../data/201302.csv')
    #df.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5']
    noise_df, noise_arr =remove_noise(df)
    ads_df, ads_index = filtering_ads(noise_df)
    nonen_df, nonen_index, lan_lst = filter_nonen(ads_df)
    final, bad_index =remove_badword(nonen_df)
    final.to_csv('../../cleandata/2013Q1.csv', sep=';')
