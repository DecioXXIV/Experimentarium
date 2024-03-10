import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, homogeneity_score, completeness_score, v_measure_score

### ################## ###
### PRINCIPAL FUNCTION ###
### ################## ###

def evaluate_experiments(cl_models, tfidf_vectors, bert_vectors, dataset):
    columns = ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score', 'Homogeneity Score']
    index = ['KMeans-TFIDF', 'KMeans-BERT', 'Agg-TFIDF', 'Agg-BERT']
    df = pd.DataFrame(index=index, columns=columns)

    true_labels = list()
    for i in range(0, len(dataset)):
        row = dataset.iloc[i]
        label = np.where(row == 1)[0][0]
        true_labels.append(label)

    # KMeans-TFIDF
    model = cl_models[0]
    labels = model.labels_
    sil_score = silhouette_score(tfidf_vectors, labels)
    db_score = davies_bouldin_score(tfidf_vectors, labels)
    ch_score = calinski_harabasz_score(tfidf_vectors, labels)
    hm_score = homogeneity_score(true_labels, labels)

    df.at['KMeans-TFIDF', 'Silhouette Score'] = sil_score
    df.at['KMeans-TFIDF', 'Davies-Bouldin Score'] = db_score
    df.at['KMeans-TFIDF', 'Calinski-Harabasz Score'] = ch_score
    df.at['KMeans-TFIDF', 'Homogeneity Score'] = hm_score

    # KMeans-BERT
    model = cl_models[1]
    labels = model.labels_
    sil_score = silhouette_score(bert_vectors, labels)
    db_score = davies_bouldin_score(bert_vectors, labels)
    ch_score = calinski_harabasz_score(bert_vectors, labels)
    hm_score = homogeneity_score(true_labels, labels)

    df.at['KMeans-BERT', 'Silhouette Score'] = sil_score
    df.at['KMeans-BERT', 'Davies-Bouldin Score'] = db_score
    df.at['KMeans-BERT', 'Calinski-Harabasz Score'] = ch_score
    df.at['KMeans-BERT', 'Homogeneity Score'] = hm_score
    
    # Agg-TFIDF
    model = cl_models[2]
    labels = model.labels_
    sil_score = silhouette_score(tfidf_vectors, labels)
    db_score = davies_bouldin_score(tfidf_vectors, labels)
    ch_score = calinski_harabasz_score(tfidf_vectors, labels)
    hm_score = homogeneity_score(true_labels, labels)

    df.at['Agg-TFIDF', 'Silhouette Score'] = sil_score
    df.at['Agg-TFIDF', 'Davies-Bouldin Score'] = db_score
    df.at['Agg-TFIDF', 'Calinski-Harabasz Score'] = ch_score
    df.at['Agg-TFIDF', 'Homogeneity Score'] = hm_score

    # Agg-BERT
    model = cl_models[3]
    labels = model.labels_
    sil_score = silhouette_score(bert_vectors, labels)
    db_score = davies_bouldin_score(bert_vectors, labels)
    ch_score = calinski_harabasz_score(bert_vectors, labels)
    hm_score = homogeneity_score(true_labels, labels)

    df.at['Agg-BERT', 'Silhouette Score'] = sil_score
    df.at['Agg-BERT', 'Davies-Bouldin Score'] = db_score
    df.at['Agg-BERT', 'Calinski-Harabasz Score'] = ch_score
    df.at['Agg-BERT', 'Homogeneity Score'] = hm_score
    
    print("Results report\n")
    print(df)

    return df