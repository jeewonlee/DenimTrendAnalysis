import pickle
import pandas as pd
import datetime

if __name__ == '__main__':
    mat = pickle.load(open("../../picklefiles/nmf.pkl", "rb"))
    # print mat
    # print mat.shape
    # print type(mat)
    df = pd.read_csv('../../cleandata/all.csv', delimiter=';')
    if len(df.columns)==6:
        df.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5']
    elif len(df.columns)==7:
        df.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5','yo']
    else:
        print 'too many columns', len(df.columns)
    print "length of dataframe",len(df)
    print df.date.value_counts()
    print type(df.date[100])
    #dates = datetime.datetime.strptime(df.date,'%Y-%m-%d')
    df['date'] = pd.to_datetime(df['date'])
    #dates = df['date'].apply(datetime.datetime)

    dates = df['date']
    print dates
    print type(dates)
    mat_df = pd.DataFrame(mat)
    mat_df.set_index(dates)
    #print mat_df
