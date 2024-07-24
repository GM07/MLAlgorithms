import unittest
import numpy as np
from mlalgorithms.clustering.kmeans import KMeans

class KMeansTest(unittest.TestCase):
    def setUp(self):
        self.data = [
            [1.0,   1.0],
            [5.0,   0.0],
            [2.0,   1.0],
            [5.5,   0.0],
            [-10.0, 0.0],
        ]
