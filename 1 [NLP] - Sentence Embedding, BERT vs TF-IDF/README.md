# Sentence Embedding, BERT vs TF-IDF

In this experiment, we test a technique to accomplish the Embedding of Sentences, meant as strings composed by more than a single token.

- Baseline: TF-IDF vectorization.
- Proposed Technique: sentence vectorization with pre-trained BERT.

The evaluation of the outcome is performed by clustering the vectors extracted from 2700 sentences (belonging to 4 classes: "scientific paper title", "recipe", "product description", "song text") with the Algorithms [KMeans++](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and [Agglomerative Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html). 

The goodness of the outcome is related to the following 4 chosen metrics for Clustering evaluation:
- [Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) (to be maximized)
- [Davies-Bouldin Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html) (to be minimized)
- [Calinski-Harabasz Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html) (to be maximized)
- [Homogeneity Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html) (to be maximized)

## Outcome (the best score for each Metric is <ins>underlined</ins>)
| | Silhouette Score | Davies-Bouldin Score | Calinski-Harabasz Score | Homogeneity Score | Training Time (iterations) |
| ---------- | ---------------- | -------------------- | ----------------------- | ----------------- | --------------- |
| **KMeans-TFIDF** | 0.018282 | 6.946187 | 29.841708 | **<ins>0.629896</ins>** | 0:00:50.746153 (25) |
| **KMeans-TFIDF** | **<ins>0.280503</ins>** | **<ins>1.363238</ins>** | **<ins>1094.778439</ins>** | 0.414508 | 0:00:05.596511 (25) |
| **Agg-BERT** | 0.021381 | 4.734103 | 29.883223 | 0.578523 | 0:00:16.580724 (1) |
| **Agg-BERT** | 0.263468 | 1.37603 | 988.543754 | 0.545182 | 0:00:00.949103 (1) |

### Observations
- The outcome provided from Clustering on BERT vectors is better than the one provided from Clustering on TF-IDF vectors on all the metrics, but not on the Homogeneity Score.
- Globally, "Agg-BERT" appears as the best pipeline: 2nd best Silhouette Score, 2nd best DB-Score, 2nd best CH-Score, 3rd best HM-Score, 1st best Trainig Time.
- Silhouette Score, DB-Score and CH-Score suggest that working on BERT Vectors leads to a better Clustering. Despite this fact, the Homogeneity Score gets worse.