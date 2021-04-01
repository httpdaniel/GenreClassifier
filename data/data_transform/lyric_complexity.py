# Import packages
import pandas as pd
import nltk as nlp
from nltk.tokenize import word_tokenize

# Read in csv
df = pd.read_csv('style.csv')

# Lyric count
df['Lyric_Count'] = df.Lyrics.apply(lambda x: len(str(x).split(' ')))

# Character count
df['Character_Count'] = df.Lyrics.apply(lambda x: len(str(x)))

# POS count
df['Noun_Count'] = df.Nouns.apply(lambda x: len(str(x).split(' ')))
df['Verb_Count'] = df.Verbs.apply(lambda x: len(str(x).split(' ')))
df['Adjective_Count'] = df.Adjectives.apply(lambda x: len(str(x).split(' ')))
df['Adverb_Count'] = df.Adverbs.apply(lambda x: len(str(x).split(' ')))


# Type token ratio
def ttr(text):
    word_tokens = word_tokenize(text)
    types = nlp.Counter(word_tokens)
    ratio = (len(types)/len(word_tokens))*100
    return round(ratio, 2)


df['TTR'] = df.Lyrics.apply(ttr)

df.to_csv('complexity.csv', index=False, encoding='utf-8')
