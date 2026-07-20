# run from project root
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weighting_scheme.functions import search

# read data
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
name_to_id = pd.read_csv(DATA / 'name_to_id.csv').set_index('name')
id_to_name = name_to_id.reset_index().set_index("id")

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
