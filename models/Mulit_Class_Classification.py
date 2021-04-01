#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
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

from sklearn.model_selection import cross_val_score

import seaborn as sns


# In[2]:


df = pd.read_csv('dataset.csv')
df.head()


# In[3]:


X = df.iloc[:,4].values.astype('U')
for i in range(5545):
    X[i] = str.lower(X[i])


# In[4]:


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


# In[5]:


d = {'Genre' : df['Genre'] , 'Lyrics' : X , 'Genre_ID' : y_output}
df2 = pd.DataFrame(data = d)


# In[6]:


df2.head()


# In[7]:


fig = plt.figure(figsize=(8,6))
df2.groupby('Genre').Lyrics.count().plot.bar(ylim=0)
plt.show()


# In[8]:


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')


# In[9]:


features = tfidf.fit_transform(df2.Lyrics).toarray()
labels = df2.Genre_ID
features.shape


# In[10]:


N = 2
for genre , genre_id in zip(list(df2['Genre'].drop_duplicates()) , list(df2['Genre_ID'].drop_duplicates()) ):
    features_chi2 = chi2(features, labels == genre_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(genre))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(df2['Lyrics'], df2['Genre'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# ## Cross Val

# In[36]:


#from sklearn.model_selection import GridSearchCV


# In[ ]:


#parameters = {'n_estimators' : [int((len(X)+5)/10*i) for i in range(1,10)] }
#rfc = RandomForestClassifier()

#CV_rfc = GridSearchCV(estimator=rfc, param_grid=parameters, cv= 10)
#CV_rfc.fit(X_train_tfidf, y_train)


# In[40]:


#parameters = {'C' : [1e-5 , 1e-3 , 1e-1 , 1 , 10 , 100 , 1000] }
#lin_svc = LinearSVC()

#CV_lin_svc = GridSearchCV(estimator=lin_svc, param_grid=parameters, cv= 10)
#CV_lin_svc.fit(X_train_tfidf, y_train)


# In[58]:


#parameters = {'C' : [1e-5 , 1e-3 , 1e-1 , 1 , 10 , 100 , 1000] }
#log_reg = LogisticRegression()

#CV_log_reg = GridSearchCV(estimator=log_reg, param_grid=parameters, cv= 10)
#CV_log_reg.fit(X_train_tfidf, y_train)
#print(CV_log_reg.best_estimator_)


# In[12]:


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(C=0.1),
    MultinomialNB(),
    LogisticRegression(C=1),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


# In[13]:


cv_df.groupby('model_name').accuracy.mean()


# In[14]:


model = LogisticRegression()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
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


# In[58]:


importance = sorted(model.coef_[0])[-10:]
label_rock = []
for item in importance : 
    idx = np.where(model.coef_[0] == item)
    label_rock.append(feature_names[idx][0])
plt.bar([x for x in range(len(importance))], importance)
plt.xticks([x for x in range(len(importance))] , label_rock, rotation='vertical')
plt.title('Rap Feature Importance')
plt.show()


# In[59]:


importance = sorted(model.coef_[1])[-10:]
label_rock = []
for item in importance : 
    idx = np.where(model.coef_[1] == item)
    label_rock.append(feature_names[idx][0])
plt.bar([x for x in range(len(importance))], importance)
plt.xticks([x for x in range(len(importance))] , label_rock, rotation='vertical')
plt.title('R&B Feature Importance')
plt.show()


# In[60]:


importance = sorted(model.coef_[2])[-10:]
label_rock = []
for item in importance : 
    idx = np.where(model.coef_[2] == item)
    label_rock.append(feature_names[idx][0])
plt.bar([x for x in range(len(importance))], importance)
plt.xticks([x for x in range(len(importance))] , label_rock, rotation='vertical')
plt.title('Blues Feature Importance')
plt.show()


# In[61]:


importance = sorted(model.coef_[3])[-10:]
label_rock = []
for item in importance : 
    idx = np.where(model.coef_[3] == item)
    label_rock.append(feature_names[idx][0])
plt.bar([x for x in range(len(importance))], importance)
plt.xticks([x for x in range(len(importance))] , label_rock, rotation='vertical')
plt.title('Country Feature Importance')
plt.show()


# In[62]:


importance = sorted(model.coef_[4])[-10:]
label_rock = []
for item in importance : 
    idx = np.where(model.coef_[4] == item)
    label_rock.append(feature_names[idx][0])
plt.bar([x for x in range(len(importance))], importance)
plt.xticks([x for x in range(len(importance))] , label_rock, rotation='vertical')
plt.title('EDM Feature Importance')
plt.show()


# In[63]:


importance = sorted(model.coef_[5])[-10:]
label_rock = []
for item in importance : 
    idx = np.where(model.coef_[5] == item)
    label_rock.append(feature_names[idx][0])
plt.bar([x for x in range(len(importance))], importance)
plt.xticks([x for x in range(len(importance))] , label_rock, rotation='vertical')
plt.title('Rap Feature Importance')
plt.show()


# In[64]:


importance = sorted(model.coef_[6])[-10:]
label_rock = []
for item in importance : 
    idx = np.where(model.coef_[6] == item)
    label_rock.append(feature_names[idx][0])
plt.bar([x for x in range(len(importance))], importance)
plt.xticks([x for x in range(len(importance))] , label_rock, rotation='vertical')
plt.title('Pop Feature Importance')
plt.show()


# In[29]:


from sklearn import metrics
clf_report = metrics.classification_report(y_test, y_pred, target_names=df['Genre'].unique() , output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)

