# MLAlgorithms
Implementation of machine learning algorithms from scratch for learning purposes. PyTorch is the only dependency. Scikit-learn and Matplotlib are used in the `example.py` file to compare the scikit-learn implementations with the custom made implementations and give examples on how to use the components. However, the library itself only depends on PyTorch.

# Requirements
- torch (2.4.0)
- [Optional] : scikit-learn (1.5.1)

# Installation
Clone the project and use algorithms. The `example.py` file shows how to use the algorithms.

```bash
pip install torch==2.4.0

git clone git@github.com:GM07/MLAlgorithms.git
cd MLAlgorithms
```

# Algorithms implemented 

## Regression
- [X] Linear Regression
- [X] Ridge Regression
- [X] Lasso Regression
- [X] Logistic Regression

## Clustering
- [X] KMeans

<p align="center">
<img src="images/kmeans.png" alt="K-means Clustering" width="400" height="400">
</p>

- [X] DBSCAN

<p align="center">
<img src="images/dbscan.png" alt="DBSCAN Clustering" width="400" height="400">
</p>

## Naive Bayes
- [X] Gaussian Naive Bayes
- [X] Bernoulli Naive Bayes
- [X] Multinomial Naive Bayes

## Dimensionality Reduction
- [X] PCA
- [X] t-SNE
- [ ] UMAP

## Decision Trees
- [X] Decision Tree Classifier
- [X] Decision Tree Regressor

## Ensemble Learning
- [ ] Bagging
- [ ] Boosting
- [ ] XGBoost
- [ ] Random Forests

## Deep Learning Layers
- [X] Linear Layer
- [X] Attention Layer
- [ ] Convolutional Layer
- [ ] Max Pooling Layer
- [ ] Average Pooling Layer
- [X] Layer Normalization Layer
- [X] Batch Normalization Layer

## Deep Learning
- [X] Neural Network
- [X] LSTM
- [ ] RNN
- [ ] Transformer
- [ ] Diffusion Model

## Metrics
- [X] RMSE

# References
- [Scikit-learn](https://scikit-learn.org/stable): Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
