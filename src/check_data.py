import pandas as pd
import numpy as np

if __name__ == '__main__':
    print "2013 Q1"
    df = pd.read_csv('../../cleandata/2013Q1.csv', sep=';')
    print df.columns
    print len(df)
    print df.date.unique()
    print "2013 Q2"
    df = pd.read_csv('../../cleandata/2013Q2.csv', sep=';')
    print df.columns
    print len(df)
    print df.date.unique()
    print "2014 Q2"
    df = pd.read_csv('../../cleandata/2014Q2.csv', sep=';')
    print df.columns
    print len(df)
    print df.date.unique()
    print "2015 Q2"
    df = pd.read_csv('../../cleandata/2015Q2.csv', sep=';')
    print df.columns
    print len(df)
    print df.date.unique()
