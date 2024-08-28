import numpy as np
import torch

def load_classification_data(return_pt = False):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X, Y = load_iris(return_X_y=True)
    a, b, c, d = train_test_split(X, Y, test_size=0.01, random_state=0)
    if return_pt:
        return torch.Tensor(a), torch.Tensor(b), torch.tensor(c, dtype=torch.int64), torch.tensor(d, dtype=torch.int64)
    return a, b, c, d

def load_dimensionality_reduction_data():
    from sklearn.datasets import load_digits
    return load_digits(return_X_y=True)

def load_binary_classification_data():
    from sklearn.model_selection import train_test_split
    rng = np.random.RandomState(1)
    X = rng.randint(5, size=(6, 100))
    Y = np.array([1, 2, 3, 4, 4, 5])
    return X, X[4:5], Y, Y[4:5]

def load_2_class_classification_data():
    X_train = np.array([[5,6,1,3,7,4,10,1,2,0,5,3,1,4],[1,2,0,2,3,3,9,4,4,3,6,5,3,7]]).T
    Y_train = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1])
    X_test  = np.array([[2,3,3,3,2,4],[1,1,0,7,6,5]]).T
    Y_test  = np.array([0,0,0,1,1,1])
    return X_train, X_test, Y_train, Y_test

def print_name(func):
    def wrapper(*args, **kwargs):
        print('=' * 100)
        print(f"Executing function: {func.__name__}")
        print('-' * 100)
        return_value = func(*args, **kwargs)
        print('=' * 100)
        print()
        return return_value
    return wrapper

@print_name
def kmeans():
    X_train = [
        [1, 1],
        [5, 0],
        [2, 2],
        [5.5, 0],
        [-10, -5],
    ]
    X_test = [
        [1.5, 1.5],
        [-15.0, 0.0]
    ]

    from mlalgorithms.clustering.kmeans import KMeans
    from sklearn.cluster import KMeans as KMeans2
    
    clf = KMeans2(n_clusters=3, random_state=0, n_init='auto')
    clf.fit(torch.tensor(X_train))
    print('SKLEARN : ', clf.cluster_centers_)

    kmeans = KMeans(3, 5)
    kmeans.fit(torch.tensor(X_train))
    print('MLALGORITHMS : ', kmeans.clusters_centroids)

@print_name
def dbscan():

    from mlalgorithms.clustering.dbscan import DBSCAN
    import matplotlib.pyplot as plt
    from sklearn import datasets

    n_samples = 500
    seed = 30
    X, _ = datasets.make_circles(
        n_samples=n_samples, 
        factor=0.5, 
        noise=0.05, 
        random_state=seed
    )
    X = torch.tensor(X)
    dbscan = DBSCAN(5, 0.2)
    predictions = dbscan.predict(X)
    
    colors = ['#377eb8' if p == 0 else '#ff7f00' for p in predictions]
    plt.scatter(X[:, 0], X[:, 1], s=10, c=colors)
    plt.show()

@print_name
def ridge_regression():
    X = torch.Tensor([[0,0], [1, 1], [2, 2]])
    Y = torch.Tensor([0, 1, 2])

    # Custom implementation
    from mlalgorithms.supervised.regression import RidgeRegression
    from sklearn.linear_model import RidgeClassifier
    clf = RidgeClassifier(alpha=0.5)
    clf.fit(X, Y)

    reg = RidgeRegression(regularization_coef=0.5)
    reg.fit(X, Y)

    print('EXPECTED \t: \t', Y)
    print('SKLEARN \t: \t', clf.predict(X))
    print('MLALGORITHMS \t: \t', reg.predict(X))


@print_name
def lasso_regression():

    X = torch.Tensor([[0,0], [1, 1], [2, 2]])
    Y = torch.Tensor([0, 1, 2])

    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.1)
    from mlalgorithms.supervised.regression import LassoRegression
    mla = LassoRegression(regularization_coef=0.01, nb_epochs=500)
    clf.fit(X, Y)
    mla.fit(X, Y)
    print('EXPECTED \t: \t', Y)
    print('SKLEARN \t: \t', clf.predict(X))
    print('MLALGORITHMS \t: \t', mla.predict(X).squeeze())

@print_name
def logistic_regression():
    X_train, X_test, Y_train, Y_test = load_2_class_classification_data()
    print('EXPECTED \t: \t', Y_test)

    from sklearn.linear_model import LogisticRegression as LR
    clf = LR(penalty=None)
    clf.fit(X_train, Y_train)
    print('SKLEARN \t: \t', clf.predict(X_test))

    from mlalgorithms.supervised.regression import LogisticRegression
    mla = LogisticRegression()
    mla.fit(torch.Tensor(X_train), torch.Tensor(Y_train))
    print('MLALGORITHMS \t: \t', mla.predict(torch.Tensor(X_test)).squeeze())
    

@print_name
def naive_bayes():
    X_train, X_test, Y_train, Y_test = load_classification_data()
    print('EXPECTED \t: \t', Y_test)

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, Y_train).predict(X_test)    

    print('SKLEARN \t: \t', y_pred)

    from mlalgorithms.supervised.naive_bayes import GaussianNaiveBayes
    gnb2 = GaussianNaiveBayes()
    y_pred_2 = gnb2.fit(torch.tensor(X_train), torch.tensor(Y_train)).predict(torch.tensor(X_test))
    print('MLALGORITHMS \t: \t', y_pred_2)
    
@print_name
def bernoulli_naive_bayes():
    X_train, X_test, Y_train, Y_test = load_binary_classification_data()
    print('EXPECTED \t: \t', Y_test)

    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB()
    clf.fit(X_train, Y_train)
    print('SKLEARN \t: \t', clf.predict(X_test))

    from mlalgorithms.supervised.naive_bayes import BernoulliNaiveBayes
    bnb = BernoulliNaiveBayes()
    bnb.fit(torch.tensor(X_train), torch.tensor(Y_train))
    print('MLALGORITHMS \t: \t', bnb.predict(torch.tensor(X_test, dtype=torch.float32)))

@print_name
def multinomial_naive_bayes():
    X_train, X_test, Y_train, Y_test = load_classification_data()
    print('EXPECTED \t: \t', Y_test)

    from sklearn.naive_bayes import MultinomialNB
    from mlalgorithms.supervised.naive_bayes import MultinomialNaiveBayes
    clf = MultinomialNB()
    clf.fit(X_train, Y_train)
    print('SKLEARN \t: \t', clf.predict(X_test))

    bnb = MultinomialNaiveBayes(alpha_smoothing=1.0)
    bnb.fit(torch.tensor(X_train), torch.tensor(Y_train))
    print('MLALGORITHMS \t: \t', bnb.predict(torch.tensor(X_test, dtype=torch.float32)))

@print_name
def pca():
    import matplotlib.pyplot as plt
    X, Y = load_dimensionality_reduction_data()

    from mlalgorithms.dimensionality_reduction.pca import PCA
    pca = PCA(nb_dims=2)
    res = pca.predict(torch.Tensor(X))
    
    plt.scatter(res[:, 0], res[:, 1], s=20, c=Y)
    plt.show()

@print_name
def tSNE():
    from mlalgorithms.dimensionality_reduction.t_sne import tSNE
    import matplotlib.pyplot as plt
    X, Y = load_dimensionality_reduction_data()

    t_sne = tSNE(nb_dims=2, learning_rate=200, perplexity=40)
    res = t_sne.predict(torch.Tensor(X))
    plt.scatter(res[:, 0], res[:, 1], s=20, c=Y)
    plt.show()

@print_name
def nn():
    torch.random.manual_seed(42)
    X_train, X_test, Y_train, Y_test = load_classification_data(return_pt=True)
    Y_train = torch.nn.functional.one_hot(Y_train).type(dtype=torch.float32)

    from mlalgorithms.deep_learning.neural_network import NeuralNetwork
    from mlalgorithms.deep_learning.deep_model import TrainingConfig
    nn = NeuralNetwork(4, 3, hidden_sizes=[3], hidden_activations=['relu'])
    training_config = TrainingConfig(batch_size=256, device='mps', epochs=500)

    nn.fit(X_train, Y_train, training_config)
    results = nn.predict(X_test, training_config)
    results = torch.argmax(results, dim=1).detach().cpu()
    print(f'Accuracy : {100.0 * (results == Y_test.detach().cpu()).sum(dim=0) / len(Y_test)} %')

if __name__ == "__main__":
    
    # kmeans()
    # dbscan()

    # naive_bayes()
    # bernoulli_naive_bayes()
    # multinomial_naive_bayes()

    # ridge_regression()
    # lasso_regression()
    # logistic_regression()
    
    # pca()
    # tSNE()

    # nn()
