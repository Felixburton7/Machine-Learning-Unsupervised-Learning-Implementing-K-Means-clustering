

## Unsupervised Learning: Implementing K-means for Text Data Clustering and Evaluation with Wikipedia Articles

### Overview

This project showcases the implementation of the k-means algorithm, a core unsupervised learning technique, specifically applied to clustering text documents from Wikipedia. Unlike supervised learning, which depends on labeled data, unsupervised learning involves discovering hidden patterns or intrinsic structures within unlabeled data. The k-means algorithm is particularly effective for clustering, grouping data points into distinct clusters based on their similarities.

The project involved several key components:

- **Text Preprocessing:** Converting text documents into numerical feature vectors using techniques like TF-IDF.
- **Algorithm Implementation:** Developing the k-means algorithm from scratch, covering initialization, assignment, and update steps.
- **Experimental Analysis:** Exploring the effects of random initialization, varying the number of clusters, and evaluating clustering results both quantitatively and qualitatively.

### Technology Used

To accomplish this project, I used the following technologies:

- **Python 3.x**: The primary programming language for the implementation.
- **NumPy**: For numerical computations and array manipulations.
- **Pandas**: For data loading and manipulation.
- **Matplotlib**: For visualizing data and clustering results.
- **scikit-learn**: For utilities such as `TfidfVectorizer` and `pairwise_distances`.

### Steps to Implement k-means

1. **Setup and Requirements**

   - Ensure you have Python 3.x installed.
   - Install the required Python libraries using pip:

     ```bash
     pip install numpy pandas matplotlib scikit-learn
     ```

2. **Running the Notebook**

   - Clone the repository:

     ```bash
     git clone <repository-url>
     cd <repository-directory>
     ```

   - Start Jupyter Notebook:

     ```bash
     jupyter notebook
     ```

   - Open the `Assignment6_KMeans_Text_Data.ipynb` file and start working on the cells.

3. **Notebook Structure**

   - **Initial Setup**: Import the necessary libraries such as numpy, pandas, matplotlib, and scikit-learn.

     ```python
     import matplotlib.pyplot as plt
     import numpy as np
     import pandas as pd

     %matplotlib inline
     ```

   - **Load Data and Extract Features**: Load the Wikipedia dataset and extract TF-IDF features from the text data.

     ```python
     wiki = pd.read_csv('people_wiki.csv')
     wiki.head(20)
     ```

     ```python
     from sklearn.feature_extraction.text import TfidfVectorizer

     vectorizer = TfidfVectorizer(max_df=0.95)
     tf_idf = vectorizer.fit_transform(wiki['text'])
     words = vectorizer.get_feature_names_out()
     ```

   - **Normalize Vectors**: Normalize all vectors to unit length to make Euclidean distance mimic cosine distance.

     ```python
     from sklearn.preprocessing import normalize
     tf_idf = normalize(tf_idf)
     ```

   - **Implement k-means**: Implement the k-means algorithm by defining functions for initialization, cluster assignment, and centroid revision.

     - **Initial Centroids**: Randomly choose initial centroids from the data points.

       ```python
       def get_initial_centroids(data, k, seed=None):
           if seed is not None:
               np.random.seed(seed)
           n = data.shape[0]
           rand_indices = np.random.choice(n, k)
           centroids = data[rand_indices, :].toarray()
           return centroids
       ```

     - **Assign Clusters**: Assign each data point to the closest centroid.

       ```python
       from sklearn.metrics import pairwise_distances

       def assign_clusters(data, centroids):
           distances = pairwise_distances(data, centroids, metric='euclidean')
           return np.argmin(distances, axis=1)
       ```

     - **Revise Centroids**: Update the centroids to be the mean of the assigned data points.

       ```python
       def revise_centroids(data, k, cluster_assignment):
           new_centroids = []
           for i in range(k):
               assigned_data = data[cluster_assignment == i]
               new_centroids.append(assigned_data.mean(axis=0))
           return np.array(new_centroids)
       ```

   - **Iterate**: Combine the steps in an iterative process until convergence.

     ```python
     def kmeans(data, k, max_iters=100, seed=None):
         centroids = get_initial_centroids(data, k, seed)
         for i in range(max_iters):
             cluster_assignment = assign_clusters(data, centroids)
             new_centroids = revise_centroids(data, k, cluster_assignment)
             if np.all(centroids == new_centroids):
                 break
             centroids = new_centroids
         return centroids, cluster_assignment
     ```

### Evaluation

Evaluate the clustering results both quantitatively (e.g., using inertia) and qualitatively (e.g., by examining cluster contents).

By following these steps, I successfully implemented and evaluated the k-means algorithm for clustering text data from Wikipedia. The project provided insights into the clustering process and the impact of various factors such as initialization and the number of clusters.


