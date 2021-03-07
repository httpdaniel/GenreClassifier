# Import packages
import spotipy
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()
SPOTIFY_ID = os.getenv('SPOTIFY_ID')
SPOTIFY_SECRET = os.getenv('SPOTIFY_SECRET')

client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_ID, client_secret=SPOTIFY_SECRET)
token = client_credentials_manager.get_access_token()
sp = spotipy.Spotify(auth=token)

sp.trace = True
sp.trace_out = True

# Read in CSV
df = pd.read_csv('../lastfm/lastfm.csv')


# Return artist genres from Spotify
def lookup_genres(artist):
    try:
        print(artist)
        results = sp.search(q='artist:' + artist, type='artist')
        genres = results["artists"]["items"][0]["genres"]
        return ','.join(genres)
    except:
        return "Unknown"


# Loop through artists who's tags are unknown
for i in range(len(df['Tags'])):
    if df['Tags'][i] == "Unknown":
        df['Tags'].at[i] = lookup_genres(df['Artist'][i])

# If song has no tags, remove from dataset
df['Tags'].replace('Unknown', np.nan, inplace=True)
df = df.dropna()

# Save dataframe to csv
df.to_csv('spotify.csv', index=False, encoding='utf-8')



