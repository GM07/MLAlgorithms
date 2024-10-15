from mlalgorithms.clustering.dbscan import DBSCAN
import torch

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
