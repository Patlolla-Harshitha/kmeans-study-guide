# K-Means Clustering Study Guide

## Introduction
K-Means is an unsupervised machine learning algorithm used to group similar data points into clusters without labeled data.

---

## What is Clustering?
Clustering is a technique of grouping similar data points together based on their features so that objects in the same group are more similar to each other than to those in other groups.

---

## Types of Clustering Techniques
- Partition-based clustering (K-Means)
- Hierarchical clustering
- Density-based clustering (DBSCAN)
- Model-based clustering

---

## Why K-Means is Used
- Simple and fast algorithm
- Works well on large datasets
- Efficient for well-separated clusters
- Easy to implement

---

## Working Principle of K-Means
K-Means groups data into K clusters by minimizing the distance between data points and their cluster centroid.

---

## Step-by-Step Algorithm
1. Select number of clusters (K)
2. Initialize centroids randomly
3. Assign each data point to nearest centroid
4. Recompute centroids
5. Repeat until centroids do not change

---

## Mathematical Formulation
K-Means minimizes the objective function:

$$
J = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2
$$

Where:
- K = number of clusters
- C_i = cluster i
- μ_i = centroid of cluster i
- x = data point

---

## Distance Measures Used
- Euclidean Distance (most common)
$$
\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$
- Manhattan Distance (optional alternative)
$$
|x_2 - x_1| + |y_2 - y_1|
$$

---

## Choosing the Value of K
- Domain knowledge
- Trial and error
- Elbow method
- Silhouette score (advanced)

---

## Elbow Method
Used to determine optimal K by plotting WCSS (Within Cluster Sum of Squares) against number of clusters. The "elbow point" indicates best K.

---

## Advantages
- Simple and efficient
- Works well on large datasets
- Easy to understand and implement

---

## Disadvantages
- Requires predefined K
- Sensitive to outliers
- Works only for spherical clusters
- Random initialization may affect results

---

## Applications
- Customer segmentation
- Image compression
- Document clustering
- Recommendation systems
- Market segmentation

---

## Real-World Examples
- Grouping customers based on shopping behavior
- Compressing images by reducing colors
- Grouping news articles by topic
- Social media user segmentation

---

## Comparison with Other Algorithms

### K-Means vs Hierarchical Clustering
- K-Means: Fast, requires K
- Hierarchical: Slower, no need to predefine K

### K-Means vs DBSCAN
- K-Means: Works on spherical clusters
- DBSCAN: Works on arbitrary shapes and detects noise

---

## Interview Questions

### 1. What is K-Means?
K-Means is a clustering algorithm that groups data into K clusters based on similarity.

### 2. What is centroid?
Centroid is the center point of a cluster.

### 3. How do you choose K?
Using elbow method or domain knowledge.

### 4. What is WCSS?
Within Cluster Sum of Squares used to measure compactness.

### 5. Limitations of K-Means?
Requires K beforehand and is sensitive to outliers.

---

## Conclusion
K-Means is one of the most widely used clustering algorithms due to its simplicity and efficiency. It is useful for real-world unsupervised learning problems where labeled data is not available.

---

## References
- Scikit-learn Documentation
- Machine Learning Basics (Andrew Ng)
- GeeksforGeeks K-Means Guide
- Towards Data Science Articles
