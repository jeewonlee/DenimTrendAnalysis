import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go

#Most distributed brand
most = df[' brand_name'].value_counts().head(10).to_frame()
most.columns=['brand']
labels = most.index
values = most.brand
trace=go.Pie(labels=labels,values=values)
py.iplot([trace])

def plot_price(df):
    #sns.set_style("darkgrid")
    temp = df[df['price']<300]
    temp.price.hist(bins=100)
    expensive = df[df['price']>300]
    expensive.price.hist(bins=50)
    sns.boxplot("price", data=df)
    plt.show()

def plot_graph(df)
    temp = df[df['price']<300]

# def most_distributed(df):
#     most = df[' brand_name'].value_counts().head(10).to_frame()
#     labels = most.index
#     values = most.brand_name
#     trace=go.Pie(labels=labels,values=values)
#     py.iplot([trace])

if __name__ == '__main__':
    df = pd.read_csv('../../data/Shopstyle/jeans_data.csv')
    most_distributed(df)
