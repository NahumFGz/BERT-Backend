from twython import Twython
from keys.twitter_keys import APP_KEY,ACCESS_TOKEN
from nlpmodule.tools.Preprocessing import text_preprocessing


def get_tweet(tweet_id):
    twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)
    results = twitter.lookup_status(id=tweet_id,tweet_mode='extended')

    return results[0]['full_text']