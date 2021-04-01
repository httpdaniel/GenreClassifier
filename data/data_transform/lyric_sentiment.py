# Import packages
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Read in csv
df = pd.read_csv('complexity.csv')

# Vader analyzer
analyzer = SentimentIntensityAnalyzer()

sentiment = []


# Get valence for each song
def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    sentiment.append([score['pos'], score['neg'], score['neu'], score['compound']])


df.Lyrics.apply(get_sentiment)

df2 = pd.DataFrame(sentiment)
df2.columns = ['Valence_Pos', 'Valence_Neg', 'Valence_Neu', 'Valence_Compound']

new_df = pd.concat([df, df2], axis=1)

# Export to csv
new_df.to_csv('../final_datasets/final_preprocessed.csv', index=False, encoding='utf-8')
