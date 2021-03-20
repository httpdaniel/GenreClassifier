# Import packages
import pandas as pd
import plotly.io as pio
import plotly.express as px
pio.renderers.default = "browser"
pd.options.plotting.backend = "plotly"

df = pd.read_csv('../data/selected_dataset.csv')

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
                       yaxis=dict(title_text='Lyric Length', title_font=dict(size=10), tickfont=dict(size=10)))

fig_dist.show()
