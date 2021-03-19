# Import packages
import pandas as pd
import numpy as np
import operator

# Read in csv
df = pd.read_csv('spotify/spotify.csv')

# Create empty genre column
df['Genre'] = ''

# Define genres
genres = [
    'pop',
    'rock',
    'country',
    'rap',
    'blues',
    'r&b',
    'edm',
]


# If song has no tags, replace with "Unknown"
df['Tags'].replace('', 'Unknown', inplace=True)
df['Tags'].replace(np.nan, 'Unknown', inplace=True)

# Make tags lowercase
df['Tags'].str.lower()

# Replace abstract tags with broad genre
df['Tags'].replace('dance', 'edm', inplace=True, regex=True)
df['Tags'].replace('party', 'edm', inplace=True, regex=True)
df['Tags'].replace('electronic', 'edm', inplace=True, regex=True)
df['Tags'].replace('house', 'edm', inplace=True, regex=True)
df['Tags'].replace('Disco', 'edm', inplace=True, regex=True)
df['Tags'].replace('linedance', 'country', inplace=True, regex=True)
df['Tags'].replace('drill', 'rap', inplace=True, regex=True)
df['Tags'].replace('hip-hop', 'rap', inplace=True, regex=True)
df['Tags'].replace('Hip-Hop', 'rap', inplace=True, regex=True)
df['Tags'].replace('hip hop', 'rap', inplace=True, regex=True)
df['Tags'].replace('indie', 'rock', inplace=True, regex=True)
df['Tags'].replace('alternative', 'rock', inplace=True, regex=True)
df['Tags'].replace('metal', 'rock', inplace=True, regex=True)
df['Tags'].replace('funk', 'r&b', inplace=True, regex=True)
df['Tags'].replace('soul', 'r&b', inplace=True, regex=True)
df['Tags'].replace('rnb', 'r&b', inplace=True, regex=True)
df['Tags'].replace('jazz', 'blues', inplace=True, regex=True)


# Count occurrences of genre tags and decide overall genre
for i in range(len(df['Tags'])):
    genre_count = {}
    for j in range(0, len(genres)):
        count = df['Tags'][i].count(genres[j])
        genre_count[genres[j]] = count
    if all(value == 0 for value in genre_count.values()):
        this_genre = "UnknownGenre"
    else:
        this_genre = max(genre_count.items(), key=operator.itemgetter(1))[0]
    df['Genre'].at[i] = this_genre

# Remove all songs with no genre
df['Genre'].replace("UnknownGenre", np.nan, inplace=True, regex=True)
df = df.dropna()

# Save dataframe to csv
df.to_csv('../selected_dataset.csv', index=False, encoding='utf-8')

