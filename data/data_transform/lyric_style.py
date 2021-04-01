# Import packages
import spacy
import pandas as pd

# Read in dataset
df = pd.read_csv('../preprocessed_dataset.csv')

# Load spacy model
nlp = spacy.load('en_core_web_sm')


# Get list of nouns, verbs, adjectives, and adverbs in lyrics
def pos(string, p):
    doc = nlp(string)
    return ' '.join(list(set([i.text for i in doc if i.pos_ == p])))


pos_of_interest = ['NOUN', 'VERB', 'ADJ', 'ADV']

# Loop through each PoS
for ps in pos_of_interest:
    df[ps] = df.Lyrics.map(lambda x: pos(x, ps))

# Rename columns
df.rename(columns={'NOUN': 'Nouns', 'VERB': 'Verbs', 'ADJ': 'Adjectives', 'ADV': 'Adverbs'}, inplace=True)

# Export to csv
df.to_csv('style.csv', index=False, encoding='utf-8')
