# Import packages
import os
import requests
import re
import lyricsgenius as genius
import pandas as pd
from dotenv import load_dotenv
import numpy as np

project_folder = os.path.expanduser('~/GenreClassifier')
load_dotenv(os.path.join(project_folder, '.env'))
GENIUS_TOKEN = os.getenv('GENIUS_TOKEN')

api = genius.Genius('GENIUS_TOKEN')

# read in data
df = pd.read_csv('../billboard/billboard.csv')


# Return lyrics from Genius
def lookup_lyrics(song):
    try:
        return api.search_song(song).lyrics
    except:
        return ''


# Get lyrics
df['Lyrics'] = (df['Artist']+' '+df['Title']).apply(lambda x: lookup_lyrics(x))


# Remove new lines from text
def clean_txt(song):
    song = ' '.join(song.split("\n"))
    song = re.sub("[\[].*?[\]]", "", song)
    return song


df['Lyrics'] = df['Lyrics'].apply(lambda x: clean_txt(x))

# Drop song if there are no lyrics
df['Lyrics'].replace('', np.nan, inplace=True)
df = df.dropna()

# Save dataframe to csv
df.to_csv('genius.csv', index=False, encoding='utf-8')
