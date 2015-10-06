import numpy as np

# Settings
mu1 = [0, 1, 0.5, -1.5]
cov1 = [[2, 0, 0, 0],
		[0, 2, 0, 0],
		[0, 0, 2, 0],
		[0, 0, 0, 2]]
mu2 = [1, -1, 1.5, 0.5]
cov2 = [[1.5, 0, 0, 0],
		[0, 1.5, 0, 0],
		[0, 0, 1.5, 0],
		[0, 0, 0, 1.5]]
output_filenm = 'sample_data.txt'
num_samples = 100000

# Generate data
X1 = np.random.multivariate_normal(mu1, cov1, num_samples/2)
X2 = np.random.multivariate_normal(mu2, cov2, num_samples/2)
X = np.concatenate((X1, X2), axis=0)
X = X[np.random.permutation(X.shape[0]), :]

# Save generated data to text file
np.savetxt(output_filenm, X)
