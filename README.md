# Song Recommendation System

## Overview

A follow on from Imperial College London poster project (Cluster Analysis). 

In the original project, audio features from Taylor Swift's music were analysed and clustered. The natural extension was to analyse the lyrics of those songs, allowing for a recommendation system inspired by multi-view clustering.

App link: https://song-recommendation-system-d4iswpem8dnzoxdgqgzadv.streamlit.app/

---

## Datasets

### Audio features

Source: https://www.kaggle.com/datasets/jarredpriester/taylor-swift-spotify-dataset

- Number of observations: 363 (236 used)
- Number of features: 17 (8 features used for dissimilarity matrix)

### Lyrics

Source: https://www.kaggle.com/datasets/ishikajohari/taylor-swift-all-lyrics-30-albums

Dataset contains lyrics from 64 albums as text files.

---

## Workflow

- Clean data
- Transform features/create embeddings

- Build dissimilarity matrices
- Normalise dissimilarities
- Fuse matrices using convex combination
- Build recommendation function using k-NN

---

## Dissimilarity Functions

- Euclidean Distance (audio features, lyrical clustering)
- Cosine Similarity (lyric embeddings)

---

## Summary

- Built a recommendation system with the option to weight recommendations towards lyrics or audio
- Derived and implemented an algorithm to recommend songs using previous searches
- Clustered lyrics using K-Means and Euclidean Distance, found the lyric categories described by Swift (2022).

---

## Future Improvements

- Implement a like/dislike feature for recommended songs to improve recommendations
- Use lyric clustering to improve recommendations
