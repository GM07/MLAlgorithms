from mlalgorithms.clustering.kmeans import KMeans
from mlalgorithms.distances import euclidian

import numpy as np

def kmeans():
    kmeans = KMeans(3, 5)
    data = np.array([
        [1, 1],
        [5, 0],
        [2, 2],
        [5.5, 0],
        [-10, -5],
    ])
    kmeans.fit(data)
    print(kmeans._clusters_centroids)

    predictions = kmeans.predict(np.array([
        [1.5, 1.5],
        [-15.0, 0.0]
    ]))

    print(predictions)

if __name__ == "__main__":
    kmeans()
