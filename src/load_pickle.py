import pickle
import pandas as pd

if __name__ == '__main__':
    mat = pickle.load(open("../../picklefiles/nmf.pkl", "rb"))
    print mat
    print mat.shape
    df = pd.read_csv('../../cleandata/all.csv', delimiter=';')
    if len(df.columns)==6:
        df.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5']
    elif len(df.columns)==7:
        df.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5','yo']
    else:
        print 'too many columns', len(df.columns)
    print "length of dataframe",len(df)
    print df.date.value_counts()
