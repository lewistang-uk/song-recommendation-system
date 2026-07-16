# Song Recommendation System

## Overview

A follow on from an Imperial College London poster project (Cluster Analysis). 

In the original project, audio features from Taylor Swift's music were analysed and clustered, showing that her music has variety in style. The natural extension was to analyse the lyrics of those songs, allowing for a recommendation system with an A/B tested improvement, inspired by multi-view clustering.

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

- Clean data and transform features/create embeddings
- Normalise name format in both views
- Build normalised dissimilarity matrices, fuse using convex combination
- Build recommendation function using k-NN approach
- Add weighting scheme for historical searches and A/B test 

---

## Dissimilarity Functions

Cosine similarity was used for the lyric embeddings and Euclidean distance was used for the audio features. The dissimilarity matrices were combined using a convex combination, with a parameter "alpha" allowing recommendations to be weighted more towards lyrics (0.3) or audio (0.7), or balanced (0.5).

---

## Weighting Scheme

A linear weighting scheme was designed to allow historical searches to contribute to the recommendation ranking. In addition, the current search was not allowed to appear through infinite penalisation, and past searches were slightly penalised to offer different recommendations.

A LaTeX document outlining this scheme can be found in the repository.

---

## A/B Testing

An experiment was simulated over three categories of users. Each user listened to three songs from their set of albums. If at least 3 of the top 5 recommendations were not in listening history and were in their category, then the recommendations were marked as good. A significance level of 0.05 and power of 0.8 were used. The control was the simple k-NN recommender, and the alternative was the historical weighted recommender.

Categories:

- Category 1 (country/rock): Taylor Swift, Fearless, Speak Now, Red
- Category 2 (pop): 1989, reputation, Lover, Midnights
- Category 3 (poetic/indie): folklore, evermore, TTPD

| Segment    | Control rate | Alternative rate | Raw p      | Bonferroni p | Significant? |
|------------|--------------|------------------|------------|--------------|--------------|
| Overall    | 71.3%        | 80.3%          | 1.32e-7  | 5.28e-7   | Yes          |
| Category 1 | 87.5%        | 94.5%          | 4.48e-4 | 1.79e-3  | Yes          |
| Category 2 | 52.3%        | 60.9%          | 1.19e-2  | 4.76e-2   | Yes          |
| Category 3 | 74.1%        | 85.6%          | 3.41e-5 | 1.36e-4    | Yes          |

The results were significant across all categories after multiple hypothesis testing corrections, suggesting that the weighted recommendation system should be used. However, Category 2 was close to the 0.05 threshold. With more simulations, the result could become insignificant.

---

## Summary

- Applied NLP and other preprocessing techniques to lyrics and audio features of Taylor Swift's music
- Combined multiple views using convex combinations and various dissimilarity functions
- Built a recommendation system with the option to weight recommendations towards lyrics or audio
- Derived and implemented an system to recommend songs using historical searches
- A/B tested the historical weighted recommendation system

---

## Future Improvements

- Implement a like/dislike feature for recommended songs to improve recommendations
- Consider other weighting schemes, for example, exponential decay
