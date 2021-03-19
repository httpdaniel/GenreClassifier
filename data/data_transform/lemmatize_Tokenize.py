import pandas as pd
import csv
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

column_names = ['Title', 'Artist', 'Featuring','Featured_Artist','Lyrics','Tags','Genre']
data = pd.read_csv('dataset.csv', names=column_names)
X = data.iloc[:,4].values.astype('U')

def tokenization():
    songs_tokens = []
    for i in range(1,5546):
        X[i] = str.lower(X[i])
        a = word_tokenize(X[i])
        songs_tokens.append(a)
    print(len(songs_tokens))
return song_tokens

def lemmatization():
    lemmatizer = WordNetLemmatizer()
    lemmatized_data = []
    for song_tokens in songs_tokens:
        lemmas = [lemmatizer.lemmatize(word) for word in song_tokens]
        lemmatized_data.append(lemmas)
    return lemmatized_data
