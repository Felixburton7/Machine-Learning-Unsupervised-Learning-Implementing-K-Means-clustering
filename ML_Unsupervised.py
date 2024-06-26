# Clustering Text Data with k-means

# This project focuses on implementing the k-means algorithm to cluster text documents. 
# Unlike using a pre-implemented sklearn class, this project involves creating the k-means algorithm from scratch using the `numpy` library. 
# The aim is to uncover valuable insights from a set of unlabeled documents through clustering.

# For manipulating the data arrays, the `numpy` library is utilized. 
# See the following tutorial for more information on `numpy`.


# The project involves:

# * Clustering Wikipedia documents using k-means
# * Exploring the role of random initialization on the quality of the clustering
# * Examining how results differ after changing the number of clusters
# * Evaluating clustering, both quantitatively and qualitatively

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Enable inline plotting
%matplotlib inline

# Load data and extract features
# SKIP
wiki = pd.read_csv('people_wiki.csv')
wiki.head(20)

# To work with text data, first convert the documents into numerical features using TF-IDF.
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_df=0.95)  # ignore words with very high document frequency
tf_idf = vectorizer.fit_transform(wiki['text'])
words = vectorizer.get_feature_names_out()
words

# Normalize all vectors to be unit length to mitigate the issue of different document lengths.
from sklearn.preprocessing import normalize
tf_idf = normalize(tf_idf)
tf_idf

# Implementing k-means
# Initialize centroids by randomly choosing k data points as initial centroids
def get_initial_centroids(data, k, seed=None):
    if seed is not None:  # useful for obtaining consistent results
        np.random.seed(seed)
        
    n = data.shape[0]  # number of data points
    rand_indices = np.random.choice(n, k)
    centroids = data[rand_indices, :].toarray()
    return centroids

# k-means Algorithm
# Assign each data point to the closest centroid
from sklearn.metrics import pairwise_distances

# Function to assign clusters based on the closest centroid
def assign_clusters(data, centroids):
    distances = pairwise_distances(data, centroids, metric='euclidean')
    closest_cluster = np.argmin(distances, axis=1)
    return closest_cluster

# Compute new centroids as the mean of the assigned data points
def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in range(k):
        member_data_points = data[cluster_assignment == i]
        if member_data_points.shape[0] > 0:
            centroid = member_data_points.mean(axis=0)
            new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)
    return new_centroids

# Compute the heterogeneity metric to assess convergence
def compute_heterogeneity(data, k, centroids, cluster_assignment):
    heterogeneity = 0.0
    for i in range(k):
        member_data_points = data[cluster_assignment == i]
        if member_data_points.shape[0] > 0:
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances ** 2
            heterogeneity += np.sum(squared_distances)
    return heterogeneity

# k-means main loop
def kmeans(data, k, initial_centroids, max_iter, record_heterogeneity=None, verbose=False):
    centroids = initial_centroids[:]
    prev_cluster_assignment = None
    
    for itr in range(max_iter):  
        if verbose:
            print(f'Iteration {itr}')
        
        cluster_assignment = assign_clusters(data, centroids)
        centroids = revise_centroids(data, k, cluster_assignment)
        
        if prev_cluster_assignment is not None and (prev_cluster_assignment == cluster_assignment).all():
            break
        
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)
        
        prev_cluster_assignment = cluster_assignment[:]
    
    return centroids, cluster_assignment

# Plotting the heterogeneity to observe the convergence
def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(7,4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title(f'Heterogeneity of clustering over time, K={k}')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

# Run k-means with k=3 clusters for a maximum of 400 iterations and plot the heterogeneity
k = 3
heterogeneity = []
initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, max_iter=400, record_heterogeneity=heterogeneity, verbose=True)
plot_heterogeneity(heterogeneity, k)

# Identify the largest cluster
largest_cluster = np.bincount(cluster_assignment).argmax()
print(f'The largest cluster is cluster {largest_cluster}')

# Implement k-means++ for better centroid initialization
def k_means_plus_plus_initialization(data, k, seed=None):
    if seed is not None:  # useful for obtaining consistent results
        np.random.seed(seed)
        
    centroids = np.zeros((k, data.shape[1]))
    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx,:].toarray()
    
    distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()
    
    for i in range(1, k):
        idx = np.random.choice(data.shape[0], 1, p=distances/sum(distances))
        centroids[i] = data[idx,:].toarray()
        distances = np.min(pairwise_distances(data, centroids[0:i+1], metric='euclidean'), axis=1)
    
    return centroids

# Run k-means multiple times with different initializations to get the best result
def kmeans_multiple_runs(data, k, max_iter, seeds, verbose=False):
    min_heterogeneity_achieved = float('inf')
    final_centroids = None
    final_cluster_assignment = None
    if type(seeds) == int:
        seeds = np.random.randint(low=0, high=10000, size=seeds)
    
    num_runs = len(seeds)
    
    for seed in seeds:
        initial_centroids = k_means_plus_plus_initialization(data, k, seed)
        centroids, cluster_assignment = kmeans(data, k, initial_centroids, max_iter=max_iter, verbose=verbose)
        seed_heterogeneity = compute_heterogeneity(data, k, centroids, cluster_assignment)
        
        if verbose:
            print(f'seed={seed:06d}, heterogeneity={seed_heterogeneity:.5f}')
        
        if seed_heterogeneity < min_heterogeneity_achieved:
            min_heterogeneity_achieved = seed_heterogeneity
            final_centroids = centroids
            final_cluster_assignment = cluster_assignment
    
    return final_centroids, final_cluster_assignment

# Plot heterogeneity vs. k to determine the optimal number of clusters
def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
    plt.figure(figsize=(7,4))
    plt.plot(k_values, heterogeneity_values, linewidth=4)
    plt.xlabel('K')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

all_centroids = {}
all_cluster_assignment = {}
heterogeneity_values = []
seeds = [20000, 40000, 80000]
k_list = [2, 10, 25, 50, 100]

for k in k_list:
    print(f'Running k = {k}')
    all_centroids[k], all_cluster_assignment[k] = kmeans_multiple_runs(tf_idf, k, max_iter=400, seeds=seeds, verbose=True)
    score = compute_heterogeneity(tf_idf, k, all_centroids[k], all_cluster_assignment[k])
    heterogeneity_values.append(score)

plot_k_vs_heterogeneity(k_list, heterogeneity_values)

# Visualize document clusters
def visualize_document_clusters(wiki, tf_idf, centroids, cluster_assignment, k, words, display_docs=5):
    print('=' * 90)

    for c in range(k):
        print(f'Cluster {c}  ({(cluster_assignment == c).sum()} docs)')
        idx = centroids[c].argsort()[::-1]
        for i in range(5):
            print(f'{words[idx[i]]}:{centroids[c,idx[i]]:.3f}', end=' ')
        print()
        
        if display_docs > 0:
            print()
            distances = pairwise_distances(tf_idf, centroids[c].reshape(1, -1), metric='euclidean').flatten()
            distances[cluster_assignment != c] = float('inf')
            nearest_neighbors = distances.argsort()
            for i in range(display_docs):
                text = ' '.join(wiki.iloc[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
                print(f'* {wiki.iloc[

nearest_neighbors[i]]["name"]:50s} {distances[nearest_neighbors[i]]:.5f}')
                print(f'  {text[:90]}')
                if len(text) > 90:
                    print(f'  {text[90:180]}')
                print()
        print('=' * 90)

# Visualize clusters for k=2
k = 2
visualize_document_clusters(wiki, tf_idf, all_centroids[k], all_cluster_assignment[k], k, words)

# Visualize clusters for k=10
k = 10
visualize_document_clusters(wiki, tf_idf, all_centroids[k], all_cluster_assignment[k], k, words)

# Count clusters with fewer than 44 articles for k=100
k = 100
num_small_clusters = (np.bincount(all_cluster_assignment[k]) < 44).sum()
print(f'Number of small clusters (fewer than 44 articles): {num_small_clusters}')
```

This code is structured as a project that implements the k-means clustering algorithm from scratch, clusters text data, evaluates the clustering, and visualizes the results. It covers initialization, iteration, convergence checking, and multiple runs to find the best clustering based on heterogeneity. The project concludes with visualizing the clusters and examining the distribution of cluster sizes.
