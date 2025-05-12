from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np

# Generate data and apply KMeans
X, _ = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)

# Print centroids
centroids = kmeans.cluster_centers_
print("cluster centroids are as follows", centroids)

# Predict new data point
pred, _ = make_blobs(n_samples=1, centers=1, n_features=2, random_state=1)
print("new data points", pred)
cluster = kmeans.predict(pred)
print("new clusters belong to ", cluster)

# Plotting
colors = ['red', 'green', 'blue']
markers = ['o', 's', '^']

plt.figure(figsize=(10, 6))

# Plot original clusters
for i in range(3):
    points = X[kmeans.labels_ == i]
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], marker=markers[i], label=f'Cluster {i}')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X', label='Centroids')

# Plot new data point
plt.scatter(pred[:, 0], pred[:, 1], c='cyan', s=100, edgecolor='black', marker='*', label='New Data Point')

plt.legend()
plt.title('KMeans Clustering with New Data Point')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
