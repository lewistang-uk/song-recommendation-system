import re
import pandas as pd
import numpy as np
from pathlib import Path

# read data
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

lyrics_df = pd.read_csv(DATA / 'lyrics_transformed.csv').drop('Unnamed: 0', axis=1)
audio_df = pd.read_csv(DATA / 'audio_features_transformed.csv').drop('Unnamed: 0', axis=1)

# string cleaning
def transform_name(name):
    # get rid of spaces
    name = re.sub(r" ", "", name)

    # get rid of apostrophes and +
    name = re.sub(r"['’‘+\"]", "", name)

    # change punctuation to underscore
    name = re.sub(r'[()?.,\-!&]', "_", name)

    # lowercase all
    name = name.lower()

    # remove 'bonus track'
    name = re.sub(r"_bonustrack", "", name)
    
    # remove features
    name = re.sub(r'_feat_[a-z]+_', "", name)

    # final exceptions
    name = re.sub(r"popversion_", "popversion", name)
    name = re.sub(r"atthedisco_", "", name)
    
    return name

def normalise_apostrophes(name):
    return re.sub(r"[\"’‘]", "'", name)

# for search bar
list_of_names = [normalise_apostrophes(name) for name in audio_df['name']]
list_of_names.pop(list_of_names.index('Teardrops On My Guitar - Radio Single Remix'))

# make a hashmap for names and ids
names_df = pd.DataFrame({'raw': list_of_names, 'name': [transform_name(name) for name in list_of_names]})

audio_df['name'] = audio_df['name'].apply(transform_name)
lyrics_df['name'] = lyrics_df['name'].apply(transform_name)
df = audio_df.merge(lyrics_df, on='name')
df['id'] = [i for i in range(len(df))]

id_df = df[['id', 'name']]
names_df = names_df.merge(id_df, on='name').drop('name', axis=1)

order = [(names_df['raw'][i], i) for i in range(len(names_df))]
name_to_id = dict(order)
id_to_name = dict([(j, i) for (i, j) in order])

# calculate dissimilarity matrices
lyrics_df = df[lyrics_df.columns].drop('name', axis=1)
audio_df = df[audio_df.columns].drop(['name', 'album'], axis=1)

from sklearn.metrics import pairwise_distances
D_lyrics = pairwise_distances(lyrics_df, metric='cosine')
D_audio = pairwise_distances(audio_df, metric='euclidean')

# Dividing both matrices by max will scale distances to [0, 1], since all d(x, y) are non-negative and d(x, x) = 0 for all observations
audio_max, lyrics_max = np.max(D_audio), np.max(D_lyrics)

# convert matrices to DataFrames and scale
D_audio_df = pd.DataFrame(D_audio, columns=[i for i in range(len(df))])
D_lyrics_df = pd.DataFrame(D_lyrics, columns=[i for i in range(len(df))])

D_audio_df = D_audio_df/audio_max
D_lyrics_df = D_lyrics_df/lyrics_max

# recommendation algorithms - k-NN based
def recommend(query_name, alpha=0.5, k=5):
    D = alpha * D_audio_df + (1 - alpha) * D_lyrics_df
    query_id = name_to_id[query_name]
    distances = D.loc[query_id].copy()
    distances.loc[query_id] = np.inf  # don't recommend itself
    return [id_to_name[int(x)] for x in distances.nsmallest(k).index]

def recommend_weighted(query_name: str, prev_queries: list, alpha=0.5, k=5):
    """Takes query and a list of previous queries (ordered, most recent first)"""

    D = alpha * D_audio_df + (1 - alpha) * D_lyrics_df

    # weight function for current query
    if prev_queries:
        s_current = 0.8
    else:
        s_current = 1

    query_id = name_to_id[query_name]
    distances = D.loc[query_id].copy()*s_current
    distances.loc[query_id] = np.inf  # don't recommend itself

    # define weight function for previous queries
    def W(i, S):
        return (2*(S-i))/(5*S*(S+1))
        
    for i, query in enumerate(prev_queries):
        query_id = name_to_id[query]
        query_distances = D.loc[query_id].copy() * W(i, len(prev_queries))
        query_distances.loc[query_id] += 0.5/(i+1) # make it less likely to recommend something already searched 
        distances = distances + query_distances
    return [id_to_name[int(x)] for x in distances.nsmallest(k).index]

def search(query_name: str, prev_queries: list, balance='equal', use_prev=True, k=5):
    """
    Returns top 5 recommendations given query name and list of previous queries. 
    balance takes values in {'equal', 'lyrics', 'audio'}, changing the fusion weights of the dissimilarity matrices.
    use_prev is boolean. True means previous searches are used to enhance recommendations.
    """

    balance_to_alpha = {
        'equal': 0.5,
        'lyrics': 0.3, 
        'audio': 0.7
    }

    assert type(use_prev) == bool
    if use_prev == True:
        return recommend_weighted(query_name, prev_queries, alpha=balance_to_alpha[balance], k=k)
    else:
        return recommend(query_name, alpha=balance_to_alpha[balance], k=k)

# Streamlit deployment
import streamlit as st

# track search history
def clear():
    st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []

st.title('Taylor Swift Song Recommender')

# options menu
with st.sidebar:
    st.title('Menu')
    n_results = st.slider('Number of results', 3, 20, 10)
    balance = st.selectbox('Search balance', ['equal', 'audio', 'lyrics'])
    checked = st.checkbox('Use previous queries to enhance results')
    st.button('Clear History', on_click=clear)
    with st.expander('Search History'):
        for item in reversed(st.session_state.history):
            st.markdown(f"- {item}")

# main search bar
selection = st.selectbox('Choose a song that you liked:', list_of_names)

# show results
if st.button('Search'):
    if checked:
        st.write('Recommended based on your current search and search history:')
    else:
        st.write('Recommended based on your current search:')

    recommendations = search(selection, st.session_state.history, balance=balance, use_prev=checked, k=n_results)

    st.session_state.history.append(selection)

    df = pd.DataFrame({
    "Rank": range(1, len(recommendations) + 1),
    "Recommendation": recommendations
    })
    st.dataframe(df, width='stretch', hide_index=True)
    
