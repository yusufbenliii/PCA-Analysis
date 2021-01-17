import pandas as pd
from operations import *

data = pd.read_csv("data.csv")  # reading the dataset with pandas library
original_data = data.copy()  # taking the copy of the original data to use for plotting later
data = np.array(data.drop(['Brand'], axis=1))  # dropping the Brands column from the dataset to get a matrix
sns.scatterplot(data[:, 0], data[:, 1], hue=original_data['Brand'], s=100)  # Plotting the data set
plt.title("Original Data")
plt.show()

##################################################
# Transforming data
data = fit_data(data)  # using fit_data function we are scaling the dataset
plt.title("Transformed Data")
sns.scatterplot(data[:, 0], data[:, 1], hue=original_data['Brand'], s=100)  # Plotting the scaled set
plt.show()

##################################################
# Calculation of covariance matrix, eigen value and eigen vectors
col_matrix = column_matrices(data)  # taking the column vectors from the dataset
cov_matrix = covariance_matrix(col_matrix)  # calculating the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)  # calculating the eigenvalues and eigenvectors thanks to numpy library
sorted_eig_vectors = eig_vecs[:, :2]  # taking highest two value from each eigenvector
eig_scores = dot_product(data, sorted_eig_vectors)  # taking dot product of data and highest eigen vectors

###################################################
# Eigen Vector Plots
plt.title("Eigen Vectors")
plt.xlabel("PC1")
plt.ylabel("PC2")
eig_vec_plot(eig_scores, sorted_eig_vectors[0:2, :], original_data)  # Plotting eigenvectors
plt.show()

###################################################
# Data graph's which respect to eigenvectors
col_eig_scores = column_matrices(
    eig_scores)  # to call columns separately convert to column vector (same with taking transpose of a matrix)
pca_1 = col_eig_scores["column0"]  # Here we are taking the PCA 's of the dataset
pca_2 = col_eig_scores["column1"]
# Plotting the PCA graphs for visualization
pca_data_frame = {'PC1': pca_1, 'PC2': pca_2, 'brand': original_data['Brand']}
plt.title("PC1")
sns.scatterplot(pca_data_frame['PC1'], 0, hue=pca_data_frame['brand'], s=100)
plt.show()
plt.title("PC2")
sns.scatterplot(pca_data_frame['PC2'], 0, hue=pca_data_frame['brand'], s=100)
plt.show()
