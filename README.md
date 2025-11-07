# DBSCAN

This is a personal implementation of DBSCAN originally done for coursework in
the GalvanizeU M.S. in Data Science Program in San Francisco, CA in Winter 2016.
It will occasionally be refactored for further clarity and enhancements.

## What is DBSCAN?

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised machine learning algorithm used to identify clusters in data. It groups together points that are close to each other and labels points in low-density regions as outliers.

Unlike K-Means, DBSCAN does not require specifying the number of clusters beforehand and can discover clusters of arbitrary shape.

---

##  Parameters

- **eps** → The maximum distance between two points for one to be considered as in the neighborhood of the other.
- **min_samples** → The minimum number of points required to form a dense region (a cluster).
- **data** → The dataset (usually as a NumPy array).

---

##  Example Usage

```python
from DBSCAN import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Generate sample data
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Run DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit(X)

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title("DBSCAN Clustering Example")
plt.show()
