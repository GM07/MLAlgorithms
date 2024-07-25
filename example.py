import numpy as np

def kmeans():
    from mlalgorithms.clustering.kmeans import KMeans
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

def linear():
    X = np.array([[0], [1], [2]])
    Y = np.array([1, 2, 3])

    # Custom implementation
    from mlalgorithms.supervised.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X, Y)
    print('weights : ', reg._weights)
    print('ours pred : ', reg.predict(np.array([[3]])))

if __name__ == "__main__":
    kmeans()
    linear()
