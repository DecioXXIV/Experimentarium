import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

### #################### ###
### PRUINCIPAL FUNCTIONS ###
### #################### ###

def clustering_km(dataset):
    seeds = np.random.randint(low=0, high=65536, size=25)
    best_score, best_model = -np.inf, None

    start = datetime.now()
    for s in seeds:
        model = KMeans(n_clusters=4, init='k-means++', random_state=s).fit(dataset)
        sil_score = silhouette_score(dataset, model.labels_)

        if sil_score > best_score:
            best_score = sil_score
            best_model = model
    
    end = datetime.now()
    print(" -> Required time:", str(end-start))

    return best_model
        
def clustering_agg(dataset):
    start = datetime.now()
    model = AgglomerativeClustering(n_clusters=4).fit(dataset)
    end = datetime.now()
    print(" -> Required time:", str(end-start))

    return model