# Song Recommendation System

## Overview

A follow on from an Imperial College London poster project (Cluster Analysis). 

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

- Build dissimilarity matrices using various dissimilarity functions
- Normalise dissimilarities
- Fuse matrices using convex combination
- Build recommendation function using k-NN
- Add weighting scheme for historical searches

---

## Dissimilarity Functions

Cosine similarity was used for the lyric embeddings and Euclidean distance was used for the audio features. The dissimilarity matrices were combined using a convex combination, with a parameter "alpha" allowing recommendations to be weighted more towards lyrics (0.3) or audio (0.7), or balanced (0.5).

---

## Weighting Scheme

A linear weighting scheme was designed to allow historical searches to contribute to the recommendation ranking. In addition, the current search was not allowed to appear through infinite penalisation, and past searches were slightly penalised to offer different recommendations.

A LaTeX document outlining this scheme can be found in the repository.

---

## Summary

- Applied NLP and other preprocessing techniques to lyrics and audio features of Taylor Swift's music
- Combined multiple views using convex combinations and various dissimilarity functions
- Built a recommendation system with the option to weight recommendations towards lyrics or audio
- Derived and implemented an algorithm to recommend songs using previous searches

---

## Future Improvements

- Implement a like/dislike feature for recommended songs to improve recommendations
- Consider other weighting schemes, for example, exponential decay
