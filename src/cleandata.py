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
import jw_tokenize as tw
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

def create_yr_df(filename1, filename2, filename3,filename4):
    df = pd.read_csv(filename1,delimiter=';')
    #df.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5']
    df1 = pd.read_csv(filename2,delimiter=';')
    #df1.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5']
    df2 = pd.read_csv(filename3,delimiter=';')
    #df2.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5']
    df3 = pd.read_csv(filename4,delimiter=';')
    print df.columns
    print df1.columns
    print df2.columns
    print df3.columns
    df=df.append(df1, ignore_index=True)
    df=df.append(df2, ignore_index=True)
    df=df.append(df3, ignore_index=True)
    df.to_csv('../../cleandata/all.csv', sep=';')
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
        try:
            lan = detect(str(tweet).decode("utf8"))
            print index
            if lan !='en':
                nonen_index.append(index)
                lan_list.append(lan)
        except:
            continue
    return df.drop(df.index[nonen_index]), nonen_index, lan_list

#remove badwords
def remove_badword(df):
    bad_index=[]
    for index, tweet in enumerate(df.text):
        sentence = str(tweet).decode("utf8").lower()
        if 'niggas' in sentence:
            bad_index.append(index)
    return df.drop(df.index[bad_index]), bad_index

def remove_ads_iklan(df):
    bad_index=[]
    for index, tweet in enumerate(df.text):
        sentence = str(tweet).decode("utf8").lower()
        if 'iklan' in sentence:
            bad_index.append(index)
        if '@IklanOnlineShop' in sentence:
            bad_index.append(index)
        if 'promo' in sentence:
            bad_index.append(index)
    return df.drop(df.index[bad_index]), bad_index

def remove_songs(df):
    bad_index=[]
    for index, tweet in enumerate(df.text):
        if 'everything is blue' in tweet.lower():
            bad_index.append(index)
        if 'his pills' in tweet.lower():
            bad_index.append(index)
        if 'stight jeans on' in tweet.lower():
            bad_index.append(index)
        if 'she feel my shit' in tweet.lower():
            bad_index.append(index)
        if 'let you put your hands on me in my skin tight jeans' in tweet.lower():
            bad_index.append(index)
        if 'walked into the room you know you made my eyes burn' in tweet.lower():
            bad_index.append(index)
        if 'so you can keep me inside the pocket of your ripped jeans' in tweet.lower():
            bad_index.append(index)
        if 'apple bottom jeans' in tweet.lower():
            bad_index.append(index)
        if 'boots with the fur' in tweet.lower():
            bad_index.append(index)
        if 'you still have to squeeze into your jeans' in tweet.lower():
            bad_index.append(index)
        if "but you're perfect to me" in tweet.lower():
            bad_index.append(index)
        if "she got the blue jeans painted on tight" in tweet.lower():
            bad_index.append(index)
        if "that everybody wants on a saturday night" in tweet.lower():
            bad_index.append(index)
        if "you cut those jeans just right" in tweet.lower():
            bad_index.append(index)
        if "help me take off my balmain jeans" in tweet.lower():
            bad_index.append(index)
        if "I can see your toner through those jeans" in tweet.lower():
            bad_index.append(index)
        if "that's my dick" in tweet.lower():
            bad_index.append(index)
        if "she likes hearin' how good she looks in them blue jeans" in tweet.lower():
            bad_index.append(index)
        # sentence = str(tweet).decode("utf8").lower()
        # if 'iklan' in sentence:
        #     bad_index.append(index)
        # if '@IklanOnlineShop' in sentence:
        #     bad_index.append(index)
        # if 'promo' in sentence:
        #     bad_index.append(index)
    return df.drop(df.index[bad_index]), bad_index

def remove_more_songs(df):
    bad_index=[]
    for index, tweet in enumerate(df.text):
        if 'so you can keep me in the pocket of your ripped jeans' in tweet.lower():
            bad_index.append(index)
        if 'she likes hearing how good she looks in them blue jeans' in tweet.lower():
            bad_index.append(index)
        if 'inside the pocket of your ripped jeans' in tweet.lower():
            bad_index.append(index)
        if 'she likes hearing how good' in tweet.lower():
            bad_index.append(index)
        if 'she likes hearin how good' in tweet.lower():
            bad_index.append(index)
        if 'she looks in them blue jeans' in tweet.lower():
            bad_index.append(index)
        if 'your heart racing' in tweet.lower():
            bad_index.append(index)
        if 'teenage dream tonight' in tweet.lower():
            bad_index.append(index)
        if 'she feels my shit' in tweet.lower():
            bad_index.append(index)
        if 'inside the pocket of your ripped jeans' in tweet.lower():
            bad_index.append(index)
        if 'so you can keep me' in tweet.lower():
            bad_index.append(index)
        if 'so u can keep me inside' in tweet.lower():
            bad_index.append(index)
    return df.drop(df.index[bad_index]), bad_index

def remove_more_ads(df):
    bad_index=[]
    for index, tweet in enumerate(df.text):
        if '@SaturdayOnline' in tweet.lower():
            bad_index.append(index)
        if 'romeo' in tweet.lower():
            bad_index.append(index)
        if '#saturdaynightonline' in tweet.lower():
            bad_index.append(index)
        if '#playp9romeo' in tweet.lower():
            bad_index.append(index)
    return df.drop(df.index[bad_index]), bad_index

if __name__ == '__main__':
    # df = create_df('../../data/201609.csv','../../data/201610.csv')
    # # #ads_df = pd.read_csv('../../data/201507.csv', delimiter=';')
    # # #ads_df.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5']
    # noise_df, noise_arr =remove_noise(df)
    # ads_df, ads_index = filtering_ads(noise_df)
    # nonen_df, nonen_index, lan_lst = filter_nonen(ads_df)
    # final, bad_index =remove_badword(nonen_df)
    # final.to_csv('../../cleandata/2016Q4.csv', sep=';')
    # #create_yr_df('../../cleandata/2013.csv', '../../cleandata/2014.csv', '../../cleandata/2015.csv', '../../cleandata/2016.csv')
    df = pd.read_csv('../../cleandata/all_v3.csv', delimiter=';')
    final, index = remove_more_ads(df)
    final.to_csv('../../cleandata/all_v4.csv', sep=';')
