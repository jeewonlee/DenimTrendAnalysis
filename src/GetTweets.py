import got
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# def main(query, since, until, max_tweet, filename):
# 	tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query).setSince(since).setUntil(until).setMaxTweets(max_tweet)
# 	tweet = got.manager.TweetManager.getTweets(tweetCriteria)
# 	with open(filename,'wb') as f:
# 		f.write('date;text;geo;mentions;hashtags;\n')
# 		for t in tweet:
# 			f.write(str(t.date.strftime("%Y-%m-%d"))+';'+str(t.text.replace(';',''))+';'+str(t.geo.replace(';',''))+';'+str(t.mentions.replace(';',''))+';'+str(t.hashtags.replace(';','')+';'+'\n'))
# 	f.close()
def write_file(query, since, until, max_tweet, filename):
	with open(filename,'ab') as f:
#		f.write('date;text;geo;mentions;hashtags;\n')
		for s, u in zip(since, until):
			tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query).setSince(s).setUntil(u).setMaxTweets(max_tweet)
			tweet = got.manager.TweetManager.getTweets(tweetCriteria)
			for t in tweet:
				f.write(str(t.date.strftime("%Y-%m-%d"))+';'+str(t.text.replace(';',''))+';'+str(t.geo.replace(';',''))+';'+str(t.mentions.replace(';',''))+';'+str(t.hashtags.replace(';','')+';'+'\n'))
			print tweetCriteria.since
	f.close()
if __name__ == '__main__':
		# with open(filename,'wb') as f:
		# 	f.write('date;text;geo;mentions;hashtags;\n')
	days = np.arange(1,32)
	since = []
	for day in days:
		if day<10:
			since.append('2016-10-'+'0'+str(day))
		else:
			since.append('2016-10-'+str(day))
	until = since[:]
	until.append('2016-11-01')
	until.pop(0)

	#for s, u in zip(since, until):
	write_file('jeans', since, until,1000,'../../data/twitter/201610.csv')
