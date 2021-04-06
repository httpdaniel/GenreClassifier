# Import packages
import pandas as pd
import plotly.io as pio
import plotly.express as px
import numpy as np
import tensorflow_hub as hub
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
pio.renderers.default = "browser"
pd.options.plotting.backend = "plotly"

df = pd.read_csv('../data/final_datasets/final_raw.csv')

# Plot the spread of genres in the dataset
fig = df['Genre'].value_counts().plot.bar()
fig.update_layout(
    title="Spread of genres",
    xaxis=dict(title='Genre'),
    yaxis=dict(title='Count'),
)
fig.show()

# Plot most popular artists in the dataset
fig = df['Artist'].value_counts()[:20].plot.bar()
fig.update_layout(
    title="Most popular artists",
    xaxis=dict(title="Artist"),
    yaxis=dict(title="Number of songs")
)
fig.show()

# Sunburst of artists and genres
df_sub = df[['Genre', 'Artist']]
df_sub['Count'] = 1

fig_sunburst = px.sunburst(df_sub, path=['Genre', 'Artist'], values='Count',
                           color='Genre', hover_name=None, hover_data=None)

margin_param = dict(l=25, r=25, b=50, t=50, pad=0)

fig_sunburst.update_layout(title="Genre and Artists",
                           margin=margin_param,
                           width=1000,
                           height=1000)

fig_sunburst.show()

# Plot length of lyrics
df['Lyric_Count'] = df['Lyrics'].map(lambda x: len(x.split()))

lyric_count = []

for i in df.Genre.unique():
    lyric_count.append(pd.DataFrame(df[df.Genre == i]['Lyric_Count']))

word_counts = pd.concat([i for i in lyric_count], axis=1)
word_counts.columns = df.Genre.unique()

fig_dist = word_counts.plot.box(template='ggplot2')

margin_param = dict(l=25, r=25, b=50, t=50, pad=0)

fig_dist.update_layout(title="Lyric length by genre",
                       margin=margin_param,
                       width=900,
                       height=500,
                       xaxis=dict(title_text='Genre', title_font=dict(size=10), tickfont=dict(size=10)),
                       yaxis=dict(title_text='Lyric Length (Num words)', title_font=dict(size=10),
                                  tickfont=dict(size=10)))

fig_dist.show()


# Most common words
def get_most_common(text):
    stop_words = set(stopwords.words('english'))
    vect = CountVectorizer(ngram_range=(1, 1), stop_words=stop_words)
    X = vect.fit_transform(text.values.astype('U'))
    counts = pd.DataFrame(np.asarray(X.sum(axis=0))[0], vect.get_feature_names(), columns=['count'])
    counts = counts.sort_values(by='count', ascending=False)
    return counts


# Genre map
def get_genre_map(data, pos, n):
    all_df = []
    for i in data.Genre.unique():
        temp_df = get_most_common(df[df['Genre'] == i][pos]).head(n)
        temp_df['Genre'] = i
        all_df.append(temp_df)
    all_df = pd.concat([i for i in all_df])
    all_df.reset_index(drop=False, inplace=True)
    all_df.columns = ['Word', 'Count', 'Genre']
    all_df['All_Genres'] = 'All Genres'

    treemap = px.treemap(all_df, path=['All_Genres', 'Genre', 'Word'], values='Count', )

    treemap.update_layout(title=f"Most common {pos} by genre", width=900, height=900, autosize=False, margin=dict(l=40,
                                                                                                                  r=40,
                                                                                                                  b=85,
                                                                                                                  t=100,
                                                                                                                  pad=0,
                                                                                                                  ))

    treemap.show()


get_genre_map(df, 'Nouns', 100)
get_genre_map(df, 'Verbs', 100)
get_genre_map(df, 'Adjectives', 100)
get_genre_map(df, 'Adverbs', 100)
get_genre_map(df, 'Lyrics', 100)

USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def get_embed(_string, use=USE):
    return np.array(use([_string])[0])


df_embed = pd.DataFrame([get_embed(df.Lyrics[i]) for i in range(df.shape[0])],
                        index=[df.Title[i] for i in range(df.shape[0])])

norm = [float(i)/max(df.Lyric_Count) for i in df.Lyric_Count]
df['lyric_count_norm'] = norm
df['lyric_count_norm'] = df['lyric_count_norm'].map(lambda x: x*55)


def get_pca(data, df_embedding, n_components):
    cols = df_embedding.index
    embeddings = df_embedding.iloc[:, 1:]

    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    new_values = pca.transform(embeddings)

    if n_components == 2:
        columns = ['pca_x', 'pca_y']

    elif n_components == 3:
        columns = ['pca_x', 'pca_y', 'pca_z']

    else:
        columns = ['pc']

    df_reduced = pd.DataFrame(new_values, index=cols)
    df_reduced.columns = columns

    if n_components == 1:
        df_reduced.sort_values(by='pc', ascending=False, inplace=True)

    df_merge = pd.merge(data, df_reduced, how='inner', left_on='Title', right_on=df_reduced.index)
    return df_merge


def plot_pca(data, embed, n_components):
    data = get_pca(data, embed, n_components)
    figure = go.Figure()

    genres = data.Genre.unique()

    for i in range(len(genres)):
        df_mask = data[data.Genre == genres[i]]
        df_mask['Artist_Song'] = df_mask['Artist'] + ' // ' + df_mask['Title']

        if n_components == 2:
            figure.add_trace(go.Scatter(
                x=df_mask['pca_x'],
                y=df_mask['pca_y'],
                name=genres[i],
                text=df_mask['Artist_Song'],
                mode='markers', hoverinfo='text',
                marker={'size': df_mask.lyric_count_norm}))
        else:
            figure.add_trace(go.Scatter3d(
                x=df_mask['pca_x'],
                y=df_mask['pca_y'],
                z=df_mask['pca_z'],
                name=genres[i],
                text=df_mask['Artist_Song'],
                mode='markers', hoverinfo='text',
                marker={'size': df_mask.lyric_count_norm}))

    axis_x_param = dict(showline=True,
                        zeroline=True,
                        showgrid=True,
                        showticklabels=True,
                        title='Principal Component 1')

    axis_y_param = dict(showline=True,
                        zeroline=True,
                        showgrid=True,
                        showticklabels=True,
                        title='Principal Component 2')

    legend_param = dict(bgcolor=None,
                        bordercolor=None,
                        borderwidth=None,
                        font=dict(family='Open Sans', size=15, color=None),
                        orientation='h',
                        itemsizing='constant',
                        title=dict(text='Genres (clickable!)',
                                   font=dict(family='Open Sans', size=20, color=None),
                                   side='top'), )

    figure.update_layout(legend=legend_param, title='Similarities in song lyrics by genre', width=1000, height=1000,
                         autosize=False, showlegend=True, xaxis=axis_x_param, yaxis=axis_y_param,
                         margin=dict(l=40, r=40, b=85, t=200, pad=0), )

    figure.show()
    return figure


pca_plot2 = plot_pca(df, df_embed, n_components=3)
pca_plot3 = plot_pca(df, df_embed, n_components=2)
