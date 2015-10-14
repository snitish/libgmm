import numpy as np

# Settings
mu1 = [0, 1, 0.5, -1.5]
cov1 = [[2, 0, 0, 0],
		[0, 1.5, 0, 0],
		[0, 0, 1.2, 0],
		[0, 0, 0, 2.3]]
mu2 = [1, -1, 1.5, 0.5]
cov2 = [[1.5, 0, 0, 0],
		[0, 1.8, 0, 0],
		[0, 0, 1.5, 0],
		[0, 0, 0, 1.1]]
mu3 = [-2, -1.5, 0.5, -1.5]
cov3 = [[2.1, 0, 0, 0],
		[0, 2.5, 0, 0],
		[0, 0, 1.5, 0],
		[0, 0, 0, 1.3]]
output_filenm = 'sample_data.txt'
num_samples = 300000

# Generate data
X1 = np.random.multivariate_normal(mu1, cov1, num_samples/3)
X2 = np.random.multivariate_normal(mu2, cov2, num_samples/3)
X3 = np.random.multivariate_normal(mu3, cov3, num_samples/3)
X = np.concatenate((X1, X2, X3), axis=0)
X = X[np.random.permutation(X.shape[0]), :]

# Save generated data to text file
np.savetxt(output_filenm, X)
