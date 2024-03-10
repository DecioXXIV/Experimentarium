import pandas as pd
from dataset_preprocessing import prepare_dataset
from vectorize import tfidf_vectorization, bert_vectorization
from clusterize import clustering_km, clustering_agg
from evaluate import evaluate_experiments

### #### ###
### MAIN ###
### #### ###

# FIRST STEP: Dataset loading & Preprocessing (punctuaction removal, lemmatization, stopwords removal)
print("### ############################################ ###")
print("### FIRST STEP: Dataset loading & Pre-processing ###")
print("### ############################################ ###")

dataset = pd.read_csv("benchmark_2700.csv", sep=';')
print("Dataset loaded successfully")

texts_for_tfidf = prepare_dataset(dataset, lemmatization=True, stopw_rem=True)
texts_for_bert_base = prepare_dataset(dataset, lemmatization=False, stopw_rem=False)

dataset_for_tfidf = dataset.copy()
dataset_for_tfidf['text'] = texts_for_tfidf
dataset_for_bert = dataset.copy()
dataset_for_bert['text'] = texts_for_bert_base

print("\n### End of FIRST STEP: Dataset correctly preprocessed\n")

# SECOND STEP: Document Vectorization with 2 approaches
    # 1) TF-IDF Vectorization
    # 2) Vectorization with BERT, used as Feature Extractor
print("### ################################### ###")
print("### SECOND STEP: Document Vectorization ###")
print("### ################################### ###\n")

tfidf_vectors = tfidf_vectorization(dataset_for_tfidf)
bert_vectors = bert_vectorization(dataset_for_bert)

print("### End of SECOND STEP: Document Vectorization completed\n")

# THIRD STEP: Instance clustering with KMeans and Agglomerative Clustering
print("### ######################################################################## ###")
print("### THIRD STEP: Instance Clustering with KMeans and Agglomerative Clustering ###")
print("### ######################################################################## ###\n")

print("KMeans Clustering on TF-IDF Vectors", end = "")
km_tfidf = clustering_km(tfidf_vectors)

print("KMeans Clustering on BERT Vectors", end = "")
km_bert = clustering_km(bert_vectors)

print("Agglomerative Clustering on TF-IDF Vectors", end = "")
agg_tfidf = clustering_agg(tfidf_vectors)

print("Agglomerative Clustering on BERT Vectors", end = "")
agg_bert = clustering_agg(bert_vectors)

print("\n### End of THIRD STEP: Clustering completed\n")

# FOURTH STEP: Evaluation of the performed experiments
print("### #################################################### ###")
print("### FOURTH STEP: Evaluation of the experiments performed ###")
print("### #################################################### ###\n")

cl_models = [km_tfidf, km_bert, agg_tfidf, agg_bert]
metrics = evaluate_experiments(cl_models, tfidf_vectors, bert_vectors, dataset)
metrics.to_csv("Results.csv")

print("\n### End of FOURTH STEP: Evaluation completed\n")