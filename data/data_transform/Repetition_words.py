import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.collocations import BigramCollocationFinder as bigram_collocation
from nltk.collocations import TrigramCollocationFinder as trigram_collocation
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import TrigramAssocMeasures
from nltk import corpus
import nltk
from statistics import mean
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import chi2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn import preprocessing

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

df = pd.read_csv('../preprocessed_dataset.csv')
df.head()

# Calculating number of repeated bigrams per song. Only considered bigrams of which repetition frequency is greater than 3
bigram_score = []
for i in range(len(df.index)):
    mean_pmi = 0.0
    pmi_bigram = []
    text = df["Lyrics"][i].split()
    coll_bia=bigram_collocation.from_words(text)
    coll_bia.apply_freq_filter(3)
    bigram_freq = coll_bia.ngram_fd.items()
    bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)
    bigram_score.append(len(bigramFreqTable.index.values))

# Calculating number of repeated trigrams per song. Only considered trigrams of which repetition frequency is greater than 3
trigram_score = []
for i in range(len(df.index)):
    mean_pmi = 0.0
    pmi_trigram = []
    text = df["Lyrics"][i].split()
    coll_tri=trigram_collocation.from_words(text)
    coll_tri.apply_freq_filter(3)
    trigram_freq = coll_tri.ngram_fd.items()
    trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)
    trigram_score.append(len(trigramFreqTable.index.values))  

# Calculating number of trigram triplets per song. Only considered words of which repetition frequency is greater than 4
unigram_score = []
for i in range(len(df.index)):
    token = nltk.word_tokenize(df["Lyrics"][i])
    unigrams = Counter(ngrams(token,1))
    unigram_freq =[]
    unigram_name =[]
    for item in unigrams:
        if unigrams[item] > 4:
            unigram_freq.append(unigrams[item])
            unigram_name.append(item[0])
    d = {'unigram' : unigram_name , 'freq' : unigram_freq }
    unigramFreqTable = pd.DataFrame(data=d).sort_values(by='freq', ascending=False)
    unigram_score.append(len(unigramFreqTable.index.values))

# creating output list for ML algorithms
y_output = []
for item in df['Genre'] :
    if item == 'rock' : 
        y_output.append(0)
    elif item == 'r&b' :
        y_output.append(1)
    elif item == 'blues' :
        y_output.append(2)
    elif item == 'country' :
        y_output.append(3)
    elif item == 'edm' :
        y_output.append(4) 
    elif item == 'rap' :
        y_output.append(5)
    elif item == 'pop' :
        y_output.append(6)

X = df.iloc[:,4].values.astype('U')
d = {'Genre' : df['Genre'] , 'Lyrics' : X , 'Genre_ID' : y_output}
df2 = pd.DataFrame(data = d)

# Applying Tf-IDF
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))

# Creating dataframe including number of repeated bigrams, trigrams and unigrams per song, genres, and lyrics.
d2 = {'Lyrics' : df['Lyrics'] , 'Bigram' : bigram_score , 'Trigram' : trigram_score,'Unigram' : unigram_score ,'Genre' : df['Genre']}
df3 = pd.DataFrame(data = d2)
df3.head()

#  Plot average number of repeated Bigrams per song by genre
fig = plt.figure(figsize=(8,6))
df3.groupby('Genre').Bigram.mean().plot.bar(ylim=0)
plt.title("Mean of Average Frequency of Repetition of pair of words per song by genre")
plt.show()

#  Plot average number of repeated Trigrams per song by genre
fig = plt.figure(figsize=(8,6))
df3.groupby('Genre').Trigram.mean().plot.bar(ylim=0)
plt.title("Average Frequency of Repetition of triplets of words per song by genre")
plt.show()

#  Plot average number of repeated words per song by genre
fig = plt.figure(figsize=(8,6))
df3.groupby('Genre').Unigram.mean().plot.bar(ylim=0)
plt.title("Average Frequency of Repetition of words per song by genre")
plt.show()

# Creating a combined features array and normalising
features = tfidf.fit_transform(df2.Lyrics).toarray()
print (features.shape)
features = np.column_stack((features,bigram_score,trigram_score,unigram_score))
scaler = preprocessing.MinMaxScaler()
features_scaled = scaler.fit_transform(features)

labels = df2.Genre_ID
features_scaled.shape

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(df2['Lyrics'], df2['Genre'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# Machine Algorithms

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(max_iter=5000),
    MultinomialNB(),
    LogisticRegression(random_state=0,max_iter=600),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features_scaled, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

cv_df.groupby('model_name').accuracy.mean()

model = LogisticRegression(max_iter=600)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features_scaled, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=list(df2['Genre'].drop_duplicates()), yticklabels=list(df2['Genre'].drop_duplicates()))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=df['Genre'].unique()))