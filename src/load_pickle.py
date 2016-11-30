import pickle
import pandas as pd
import datetime
import matplotlib as plt

def generate_topic_num(n):
    topic = []
    for i in np.arange(1,n+i):
        topic.append("Topic "+str(i))
    topic.append('date')
    return topic

def load_pickle_mat(filepath):
    mat = pickle.load(open(filepath, "rb"))
    return mat

def import_tweet_df():
    df = pd.read_csv('../../cleandata/all_v3.csv', delimiter=';')
    if len(df.columns)==6:
        df.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5']
    elif len(df.columns)==7:
        df.columns = [u'date', u'text', u'geo', u'mentions', u'hashtags', u'Unnamed: 5','yo']
    else:
        print 'too many columns', len(df.columns)
    print "length of dataframe",len(df)
    return df

def create_matrix_for_extract_tweet(df, mat):
    df['date'] = df.date.apply(datetime.datetime.strptime, args=('%Y-%m-%d',))
    dates = df['date']
    mat_df = pd.DataFrame(mat)
    mat_df['date']=dates
    return mat_df

def get_tweets_for_topic(df, tweet_df,num_topics):
    with open('../analysis/extract_tweets_v10.txt','ab') as f:
        for i in range(num_topics):
            if i == 0:
                index = tweet_df.iloc[:,i].argsort()[::-1][:100]
                index = list(index)
                f.write("\n##[This is for "+str(i)+"th Topic]##\n")
                # print len(index)
                # print len(df.text[index])
                for i, line in enumerate(df.text[index]):
                    # print index[i], line
                    f.writelines(str(index[i])+"\t"+line+"\n")
            else:
                index = tweet_df.iloc[:,i].argsort()[::-1][:30]
                index = list(index)
                f.write("\n##[This is for "+str(i)+"th Topic]##\n")
                for i, line in enumerate(df.text[index]):
                    f.writelines(str(index[i])+"\t"+line+"\n")
    f.close()

def create_matrix_df_for_graph(df, mat):
    df['date'] = df.date.apply(datetime.datetime.strptime, args=('%Y-%m-%d',))
    dates = df['date']
    mat_df = pd.DataFrame(mat)
    mat_df['date']=dates

    groupdate = mat_df.groupby([mat_df.date.dt.year, mat_df.date.dt.month]).sum()
    groupdate.index.names=['year','month']
    groupdate.drop(groupdate.index[0:4], inplace=True)

    date_counts = mat_df.groupby([mat_df.date.dt.year, mat_df.date.dt.month]).count()
    date_counts.index.names=['year','month']
    date_counts.drop(date_counts.index[0:4], inplace=True)

    new = groupdate/date_counts
    del new['date']
    return new

def plot_graph(new):
    ax = new.iloc[:,:5].plot()
    ax.set_xlabel('Time Periode')
    ax.set_ylabel('yay')
    ax.set_title('Denim Trend from 2012/12 to 2016/10')
    plt.legend(fontsize = 'x-small', bbox_to_anchor=(-0.2, 1.0))

if __name__ == '__main__':
    df=import_tweet_df()
    matrix = load_pickle_mat("../../picklefiles/nmf_v10.pkl")
    tweet_df = create_matrix_for_extract_tweet(df, matrix)
    get_tweets_for_topic(df, tweet_df, 30)
    # normalized_df = create_matrix_df_for_graph(matrix)
    # plot_graph(normalized_df)
    #load_pickle_mat("../../picklefiles/nmf_V2.pkl")
