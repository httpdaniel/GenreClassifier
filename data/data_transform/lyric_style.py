# Import packages
import spacy
import pandas as pd
from nltk.corpus import cmudict
from nltk.tokenize import wordpunct_tokenize

# Read in dataset
df = pd.read_csv('../preprocessed_dataset.csv')

# Load spacy model
nlp = spacy.load('en_core_web_sm')


# Get list of nouns, verbs, adjectives, and adverbs in lyrics
def pos(string, p):
    doc = nlp(string)
    return ' '.join(list(set([i.text for i in doc if i.pos_ == p])))

def get_rhyming_count(data)
    rhyming_count=[]
    for i in range(int(len(data.index))):
      text = data["Lyrics"][i]
      tokens = [wordpunct_tokenize(text)]
      punct = set(['.', ',', '!', ':', ';'])
      filtered = [[w for w in sentence if w not in punct ] for sentence in tokens]
      # last = [ sentence[len(sentence) - 1] for sentence in filtered]
      temp = set(filtered[0])
      song_words = list(temp)
      from nltk.corpus import cmudict
      syllables=[]
      word_list=[]
      for w in song_words:
          for (word, pron) in cmudict.entries():
              if  word == w and word not in word_list:
                  word_list.append(word)
                  syllables.append([word,len(pron),pron])
      count=0
      rhyme_words=[]
      for i,s in enumerate(syllables):
        if(s[0] in rhyme_words):
          continue
        else:
          for j in range(i+1,len(syllables)):
              if(s[1]==1 or syllables[j][1]==1):
                  if(s[2]==syllables[j][2]):
                      rhyme_words.append(syllables[j][0])
                      count+=1
              else:
                  temp=-1
                  for k,m in zip(syllables[j][2][::-1],s[2][::-1]):
                      if len(k)!=1 and len(m)!=1:
                          break
                      else:
                          temp+=1
                  if(s[2][-(temp+2):]==syllables[j][2][-(temp+2):]):
                      rhyme_words.append(syllables[j][0])
                      count+=1
      rhyming_count.append(count)
      return rhyming_count


pos_of_interest = ['NOUN', 'VERB', 'ADJ', 'ADV']

# Loop through each PoS
for ps in pos_of_interest:
    df[ps] = df.Lyrics.map(lambda x: pos(x, ps))

# Rename columns
df.rename(columns={'NOUN': 'Nouns', 'VERB': 'Verbs', 'ADJ': 'Adjectives', 'ADV': 'Adverbs'}, inplace=True)

# Export to csv
df.to_csv('style.csv', index=False, encoding='utf-8')
