import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Extract cluster labels
labels = [label for _, label in pseudo_labeled_data]

#  Bar Graph: Image Count per Cluster
cluster_counts = Counter(labels)
clusters = list(cluster_counts.keys())
counts = list(cluster_counts.values())

plt.figure(figsize=(8, 5))
plt.bar(clusters, counts, color='lightgreen')
plt.xlabel("Cluster ID")
plt.ylabel("Number of Images")
plt.title("Image Count per Cluster")
plt.xticks(clusters)
plt.tight_layout()
plt.savefig("cluster_bar_chart.png")
plt.show()

# Correlation Matrix of Cluster Assignments
# Create binary cluster matrix: shape (num_images, num_clusters)
num_images = len(labels)
num_clusters = max(labels) + 1
binary_matrix = np.zeros((num_images, num_clusters))
for i, label in enumerate(labels):
    binary_matrix[i, label] = 1

# Compute correlation matrix
correlation_matrix = np.corrcoef(binary_matrix.T)

plt.figure(figsize=(7, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", xticklabels=range(num_clusters), yticklabels=range(num_clusters))
plt.title("Cluster Correlation Matrix")
plt.tight_layout()
plt.savefig("cluster_correlation_matrix.png")
plt.show()
