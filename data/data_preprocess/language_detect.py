from polyglot.detect import Detector
import pandas as pd
import numpy as np

# Read in dataset
df = pd.read_csv("../selected_dataset.csv")

# Detect language of each row
for index, row in df['Lyrics'].iteritems():
    printable_str = ''.join(x for x in row if x.isprintable())
    lang = Detector(str(printable_str), quiet=True)
    df.loc[index, 'Language'] = lang.language.code

# If not english replace value for NotEnglish
for i in range(len(df['Language'])):
    if df['Language'][i] != 'en' or "Kpop" in df['Tags'][i] or "Korean" in df['Tags'][i] or "k-pop" in df['Tags'][i]\
            or "Asian" in df['Tags'][i]:
        df['Language'].at[i] = "NotEnglish"

# Remove all songs that are not in English
df['Language'].replace("NotEnglish", np.nan, inplace=True, regex=True)
df = df.dropna()

# Remove language column
df.drop('Language', axis=1, inplace=True)

df.to_csv('selected_dataset.csv', index=False, encoding='utf-8')
