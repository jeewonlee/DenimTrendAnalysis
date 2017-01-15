# DenimTrendAnalysis
Denim Trend Analysis

1. Denim Topic Analysis - using NLP technic

Discover denim trend using social network data

    a) Data Source: twitter - scraped 1000 tweets with word “jeans” per day from 12/2012 to 11/2016

    - scraped data using GetTweets.py in src file.

    - Scraped data by month first and combined all together. Collected total 842,130 data points.

    - Used some of source codes from GetOldTweets--python(Got folder in src file)

    - data files are not in GitHub


    b) Algorithms used
    : NMF, SVD, LDA and etc

2. Folder/File descriptions

  1) Analysis

    : Model output Folder

      a) analysis___.txt: Topics generated from each models

      b) extract_tweets___.txt: Sample tweets for each topics
          (ex)extract_twetts_v2.txt - sample tweets for analysis_v2.txt(NMF)

  2) Src

      a) GetTweets.py: scraped data from twitter website

      #Folder-Got: source codes from GetOldTweets--python

      b) check_data.py: check if scraped data look okay

      c) cleandata.py: combine data files and clean noise in the data

      d) jw_tokenize.py: tokenize tweets

      e) textanalysis.py: practice file for cleandata.py and topicmodeling.py

      f) topicmodeling.py: add stopwords. Run TfidfVectorizer. Run model and save it as pickle file

      g) load_pickle.py: load model pickle files to generate Topics and save to Analysis__.txt file
      (# pickle files are not saved in github). Create W matrix as dataframe format for graph.

      h) word2vec_main.py: run word2vec
