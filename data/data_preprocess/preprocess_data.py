# Import packages
import nltk
import re
import string
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read in selected dataset
df = pd.read_csv('../selected_dataset.csv')


# convert nltk tags to wordnet tags
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# remove numbers, punctuation, emojis, and make lowercase
def clean_text(text):
    remove = string.punctuation
    remove = remove.replace("!", "")
    no_numbers = re.sub(r'\d+', '', text)
    no_punctuation = "".join([char.lower() for char in no_numbers if char not in remove])

    patterns = re.compile(pattern="["u"\U0001F600-\U0001F64F"  # emoticons
                                     u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                     u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                     u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                     u"\U00002702-\U000027B0"  # emoticons
                                     u"\U0001f926-\U0001f937"  # emoticons
                                     u"\U00010000-\U0010ffff"  # emoticons
                                     u"\u2640-\u2642"  # dingbats
                                     u"\u2600-\u2B55"  # dingbats
                                     u"\u200d"  # dingbats
                                     u"\u23cf"  # dingbats
                                     u"\u23e9"  # dingbats
                                     u"\u231a"  # dingbats
                                     u"\ufe0f"  # dingbats
                                     u"\u3030"  # dingbats
                                     "]+", flags=re.UNICODE)

    return patterns.sub(r'', no_punctuation)


# perform tokenization, stopword removal and lemmatization
def process_text(text):
    text = clean_text(text)
    word_tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))

    filtered_sentence = []

    for word in word_tokens:
        if word not in stop_words:
            filtered_sentence.append(word)

    lemmatizer = WordNetLemmatizer()
    nltk_tagged = nltk.pos_tag(filtered_sentence)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmas = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmas.append(word)
        else:
            lemmas.append(lemmatizer.lemmatize(word, tag))

    sentence = " ".join(lemmas)
    return sentence


# Preprocess song lyrics and replace
df['Lyrics'] = df['Lyrics'].apply(process_text)
df.to_csv('../preprocessed_dataset.csv', index=False, encoding='utf-8')
