import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('../../cleandata/2013Q1.csv', sep=';')
    print df.columns
    print len(df)
    print df.date.unique()
