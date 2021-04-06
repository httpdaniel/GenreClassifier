# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, \
    recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score


def get_hapax(text):
    frequency = []
    word_tokens = word_tokenize(text)
    unique_words = set(word_tokens)

    for word in unique_words:
        if word_tokens.count(word) == 1:
            frequency.append(1)

    num_hapax = sum(frequency)
    hapax_score = num_hapax/len(str(text))

    return hapax_score


def get_capitals(text):
    num_caps = sum(1 for c in text if c.isupper())

    return num_caps


def get_punc(text):
    return text.count('!')


# Encode and vectorize data
def get_data():
    df = pd.read_csv('../data/final_datasets/raw_nopop.csv')

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))

    # Encode genres
    df['Genre_ID'] = df['Genre'].factorize()[0]
    encoded_data, mapping_index = df['Genre'].factorize()
    print(mapping_index)

    df['Lyrics'] = df['Lyrics'].values.astype('U')

    df['Hapax_Score'] = df['Lyrics'].apply(get_hapax)
    df['Capitals'] = df['Lyrics'].apply(get_capitals)
    df['Exclamations'] = df['Lyrics'].apply(get_punc)

    # Target
    Y = df['Genre_ID']

    # Features
    X1 = df[['Lyric_Count', 'Character_Count', 'Noun_Count', 'Verb_Count', 'Adjective_Count', 'Adverb_Count', 'TTR',
            'Bigram_Score', 'Trigram_Score', 'Unigram_Score', 'Valence_Pos', 'Valence_Neg', 'Profanity_Count',
             'Hapax_Score', 'Capitals', 'Exclamations']]

    # Vectorize lyrics
    lyric_vec = vectorizer.fit_transform(df['Lyrics']).toarray()
    print(lyric_vec.shape)

    # Dimensionality reduction
    # vec_scaled = StandardScaler(with_mean=False).fit_transform(lyric_vec)
    # svd = TruncatedSVD(n_components=50, random_state=42)
    # X_svd = svd.fit_transform(vec_scaled)

    X2 = np.column_stack((X1, lyric_vec))
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X2)

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20, random_state=42)

    return xtrain, xtest, ytrain, ytest, mapping_index


# tune hyper-parameters using cross-validation
def crossval(x, y):
    clf = LogisticRegression(penalty='l2', max_iter=100000000)
    param_grid = {'C': [0.1, 1, 5, 10]}

    gsearch = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1)
    gsearch.fit(x, y)

    res = gsearch.cv_results_
    params = gsearch.best_params_
    return res, params


def evaluate(test, pred, feature_map, total_time, modelname):
    reversefactor = dict(zip(range(7), feature_map))

    actual = np.vectorize(reversefactor.get)(test)
    predicted = np.vectorize(reversefactor.get)(pred)

    print(classification_report(actual, predicted))

    # Accuracy
    accscore = accuracy_score(actual, predicted)
    print("Accuracy Score: ", accscore, "\n")

    # Recall
    recall = recall_score(actual, predicted, average='weighted', zero_division=0)
    print("Recall: ", recall, "\n")

    # Precision
    precision = precision_score(actual, predicted, average='weighted', zero_division=0)
    print("Precision: ", precision, "\n")

    # F1 score
    f1 = f1_score(actual, predicted, average='weighted', zero_division=0)
    print("F1: ", f1, "\n")

    # Plot confusion matrix
    conf_mat = confusion_matrix(actual, predicted)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"{modelname} - Confusion Matrix\n")
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=list(feature_map), yticklabels=list(feature_map))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Print results to file
    res = {'Accuracy': round(accscore, 6),
           'Recall': round(recall, 6),
           'Precision': round(precision, 6),
           'F1': round(f1, 6),
           'Time Taken': round(total_time, 6)
           }

    res_df = pd.DataFrame([res], columns=['Accuracy', 'Recall', 'Precision', 'F1', 'Time Taken'])
    res_df.to_csv(f'../results/{modelname}_results.csv', index=False, encoding='utf-8')


X_train, X_test, y_train, y_test, mapping = get_data()

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42),
    LogisticRegression(random_state=42),
    LinearSVC(),
    DummyClassifier(strategy="most_frequent"),
]

entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

    start_time = time.time()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    end_time = time.time()
    time_taken = end_time - start_time

    evaluate(y_test, predictions, mapping, time_taken, model_name)

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
