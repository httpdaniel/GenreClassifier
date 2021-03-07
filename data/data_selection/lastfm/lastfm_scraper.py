# Import packages
import os
import pylast
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
LASTFM_KEY = os.getenv('LASTFM_KEY')
LASTFM_SECRET = os.getenv('LASTFM_SECRET')

network = pylast.LastFMNetwork(
    api_key=LASTFM_KEY,
    api_secret=LASTFM_SECRET,
)

# read in data
df = pd.read_csv('../genius/genius.csv')


# Return tags for song from LastFM
def lookup_tags(artist, song):
    try:
        track = network.get_track(artist, song)
        tags = track.get_top_tags(limit=3)
        genre = []
        for tag in tags:
            genre.append(tag.item.get_name())
        return ','.join(genre)
    except:
        try:
            artist = network.get_artist(artist)
            tags = artist.get_top_tags(limit=3)
            genre = []
            for tag in tags:
                genre.append(tag.item.get_name())
            return ','.join(genre)
        except:
            return "Unknown"


# Get tags
df['Tags'] = df.apply(lambda x: lookup_tags(x['Artist'], x['Title']), axis=1)

# If song has no tags, replace with "Unknown"
df['Tags'].replace('', 'Unknown', inplace=True)
df['Tags'].replace(np.nan, 'Unknown', inplace=True)

# Save dataframe to csv
df.to_csv('lastfm.csv', index=False, encoding='utf-8')

