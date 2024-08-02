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
    X = np.array([[0, 1], [1, 4], [2, 2]])
    Y = np.array([1, 2, 3])
    TEST = np.array([[3, 2]])

    # Custom implementation
    from mlalgorithms.supervised.regression import RidgeRegression, LinearRegression
    reg = RidgeRegression(regularization_coef=0.5)
    reg.fit(X, Y)
    print('weights : ', reg._weights)
    print('ours pred : ', reg.predict(TEST))

def naive_bayes():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB

    from mlalgorithms.supervised.naive_bayes import GaussianNaiveBayes
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    gnb2 = GaussianNaiveBayes()
    y_pred_2 = gnb2.fit(X_train, y_train).predict(X_test)

    print("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum()))
    print("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred_2).sum()))


if __name__ == "__main__":
    # kmeans()
    # linear()
    naive_bayes()
