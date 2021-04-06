# Import packages
import pandas as pd
import nltk as nlp
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder as BigramCollocation
from nltk.collocations import TrigramCollocationFinder as TrigramCollocation
from statistics import mean
from collections import Counter
from nltk.util import ngrams

# Read in csv
df = pd.read_csv('../final_datasets/final_raw.csv')

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

# Calculate repetition scores
bigram_score = []
for i in range(len(df['Genre'])):
    mean_pmi = 0.0
    pmi_bigram = []
    text = df['Lyrics'][i].split()
    coll_bia = BigramCollocation.from_words(text)
    coll_bia.apply_freq_filter(3)
    bigram_freq = coll_bia.ngram_fd.items()
    bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram', 'freq']).sort_values(by='freq',
                                                                                              ascending=False)
    if len(bigramFreqTable.index.values) != 0:
        mean_pmi = mean(bigramFreqTable["freq"])
    bigram_score.append(mean_pmi)

trigram_score = []
for i in range(len(df['Genre'])):
    mean_pmi = 0.0
    pmi_trigram = []
    text = df["Lyrics"][i].split()
    coll_tri = TrigramCollocation.from_words(text)
    coll_tri.apply_freq_filter(3)
    trigram_freq = coll_tri.ngram_fd.items()
    trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram', 'freq']).sort_values(by='freq',
                                                                                                 ascending=False)
    if len(trigramFreqTable.index.values) != 0:
        mean_pmi = mean(trigramFreqTable["freq"])
    trigram_score.append(mean_pmi)

unigram_score = []
for i in range(len(df['Genre'])):
    token = nlp.word_tokenize(df["Lyrics"][i])
    unigrams = Counter(ngrams(token, 1))
    unigram_freq = []
    unigram_name = []
    for item in unigrams:
        if unigrams[item] > 4:
            unigram_freq.append(unigrams[item])
            unigram_name.append(item[0])
    d = {'unigram': unigram_name, 'freq': unigram_freq}
    unigramFreqTable = pd.DataFrame(data=d).sort_values(by='freq', ascending=False)
    unigram_score.append(len(unigramFreqTable.index.values))

df2 = pd.DataFrame(bigram_score)
df3 = pd.DataFrame(trigram_score)
df4 = pd.DataFrame(unigram_score)
df2.columns = ['Bigram_Score']
df3.columns = ['Trigram_Score']
df4.columns = ['Unigram_Score']

df_new = pd.concat([df, df2, df3, df4], axis=1)

df_new.to_csv('complexity.csv', index=False, encoding='utf-8')
