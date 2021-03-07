# Import packages
import pandas as pd
import requests
from bs4 import BeautifulSoup


# Handle scraping requests
def _handle_scraping(request_result):
    if request_result.status_code == 200:
        html_doc = request_result.text
        soup = BeautifulSoup(html_doc, 'html.parser')
        return soup


# Store artist and song title
artist_array = []

# Scrape charts from 2010 -> 2020
for i in range(1970, 2021):
    # Append year
    website = 'https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_' + str(i)

    # Get chart table
    res = requests.get(website)
    table_class = 'wikitable sortable'
    soup = _handle_scraping(res)
    table = soup.find('table', class_=table_class)

    # Get table body
    table_body = table.find('tbody')

    # Get table rows
    rows = table_body.find_all('tr')

    for row in rows:
        try:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            artist_array.append([cols[1].replace('"', ''), cols[2]])
        except:
            pass

df = pd.DataFrame(artist_array)
df.columns = ['Title', 'Artist']

# Remove any duplicates
df = df.drop_duplicates(subset=['Title', 'Artist'], keep='first')


# Add featuring column
def featuring(artist):
    if 'featuring' in artist:
        return 1
    else:
        return 0


# Find featured artist
def featuring_substring(artist):
    if 'featuring' in artist:
        return artist.split('featuring')[1]
    else:
        return artist


# Remove featuring artist
def featuring_remove(artist):
    if 'featuring' in artist:
        return artist.split('featuring')[0]
    else:
        return artist


# Add featuring and featured_artist columns
df["Featuring"] = df.apply(lambda row: featuring(row['Artist']), axis=1)
df['Featured_Artist'] = df.apply(lambda row: featuring_substring(row['Artist']), axis=1)
df["Artist"] = df.apply(lambda row: featuring_remove(row['Artist']), axis=1)

# Save dataframe to csv
df.to_csv('billboard.csv', index=False, encoding='utf-8')
