import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.collocations import BigramCollocationFinder as bigram_collocation
from nltk.collocations import TrigramCollocationFinder as trigram_collocation
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import TrigramAssocMeasures
from nltk import corpus
import nltk
from statistics import mean

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

# Calculated bigram collocations
bigram_score = []
for i in range(len(df.index)):
    mean_pmi = 0.0
    pmi_bigram = []
    text = df["Lyrics"][i].split()
    coll_bia=bigram_collocation.from_words(text)
    coll_bia.apply_freq_filter(3)
    bigram_freq = coll_bia.ngram_fd.items()
    bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)
    if len(bigramFreqTable.index.values) != 0:
        mean_pmi=mean(bigramFreqTable["freq"])
    bigram_score.append(mean_pmi) #Use this as a feature

# Calculated trigram collocations
trigram_score = []
for i in range(len(df.index)):
    mean_pmi = 0.0
    pmi_trigram = []
    text = df["Lyrics"][i].split()
    coll_tri=trigram_collocation.from_words(text)
    coll_tri.apply_freq_filter(3)
    trigram_freq = coll_tri.ngram_fd.items()
    trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)
    if len(trigramFreqTable.index.values) != 0:
        mean_pmi=mean(trigramFreqTable["freq"])
    trigram_score.append(mean_pmi) #Use this as a feature

# Replace the features with your set of output features
# features = tfidf.fit_transform(df2.Lyrics).toarray()
# print (features.shape)
features = np.column_stack((features,bigram_score,trigram_score)) # New feature set after adding bigram and trigram collocations into the feature set. 
scaler = preprocessing.MinMaxScaler()
features_scaled = scaler.fit_transform(features)

labels = df2.Genre_ID
features_scaled.shape

#  Machine Learning Code
X_train, X_test, y_train, y_test = train_test_split(df2['Lyrics'], df2['Genre'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

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