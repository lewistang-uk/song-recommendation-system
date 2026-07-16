import re
import pandas as pd
import numpy as np
from pathlib import Path

# read data
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

D_lyrics_df = pd.read_csv(DATA / 'lyrics_dissimilarities.csv')
D_audio_df = pd.read_csv(DATA / 'audio_dissimilarities.csv')
name_to_id = pd.read_csv(DATA / 'name_to_id.csv').set_index('name')
id_to_name = name_to_id.reset_index().set_index("id")

# cast index/column names to numbers
D_audio_df.index = D_audio_df.index.astype(int)
D_audio_df.columns = D_audio_df.columns.astype(int)

D_lyrics_df.index = D_lyrics_df.index.astype(int)
D_lyrics_df.columns = D_lyrics_df.columns.astype(int)

# recommendation algorithms - k-NN based
def recommend(query_name, alpha=0.5, k=5):
    D = alpha * D_audio_df + (1 - alpha) * D_lyrics_df
    query_id = name_to_id.loc[query_name, 'id']
    distances = D.loc[query_id].copy()
    distances.loc[query_id] = np.inf  # don't recommend itself
    return [id_to_name.loc[x]['name'] for x in distances.nsmallest(k).index]

def recommend_weighted(query_name: str, prev_queries: list, alpha=0.5, k=5):
    """Takes query and a list of previous queries (ordered, most recent first)"""

    D = alpha * D_audio_df + (1 - alpha) * D_lyrics_df

    # weight function for current query
    if prev_queries:
        s_current = 0.8
    else:
        s_current = 1

    query_id = name_to_id.loc[query_name, 'id']
    distances = D.loc[query_id].copy()*s_current
    distances.loc[query_id] = np.inf  # don't recommend itself

    # define weight function for previous queries
    def W(i, S):
        return (2*(S-i))/(5*S*(S+1))
        
    for i, query in enumerate(prev_queries):
        query_id = name_to_id.loc[query, 'id']
        query_distances = D.loc[query_id].copy() * W(i, len(prev_queries))
        query_distances.loc[query_id] += 1/(i+1) # make it less likely to recommend something already searched 
        distances = distances + query_distances
    return [id_to_name.loc[x]['name'] for x in distances.nsmallest(k).index]

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
selection = st.selectbox('Choose a song that you liked:', list(name_to_id.index))

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
