import tweepy
import configparser
from util import save_pickle
def get_dat():
    try:
        config = configparser.ConfigParser()
        config.read("tweepy_config")
        auth = tweepy.OAuthHandler(config['s1']['key'], config['s1']['secret'])
        auth.set_access_token(config['s1']['token'], config['s1']['token_secret'])
        api = tweepy.API(auth)
        tweets = [(tweet.author.screen_name,tweet.text) for tweet in api.search(q='bitcoin', count=300, lang='en')]
        save_pickle({'dat':tweets},"new.pkl")
        return tweets
    except KeyError:
        print('Extraction of tweets did not work')

if __name__ == '__main__':
    get_dat()
