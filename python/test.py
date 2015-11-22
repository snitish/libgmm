import gmm
import sklearn.mixture
import numpy as np
import time

# Create data
X = np.loadtxt('../sample_data.txt')

# Train GMM using libgmm
st = time.time()
g1 = gmm.GMM(k=3, CovType='diagonal', InitMethod='kmeans')
g1.fit(X)
llh = g1.score(X)
print 'Score = ' + str(llh)
en = time.time()
print 'time1 = ' + str(en-st) + ' s'

# Train GMM using sklearn GMM
st = time.time()
g2 = sklearn.mixture.GMM(n_components=3, covariance_type='diag', tol=1e-6, verbose=1)
g2.fit(X)
en = time.time()
print 'time2 = ' + str(en-st) + ' s'
