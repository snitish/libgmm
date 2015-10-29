# libgmm
A library for training Gaussian Mixture Models written in C.

### How to build
- To enable OpenMP, add the '-fopenmp' option to line 2 of Makefile

 ```
 CFLAGS = -std=c99 -O3 -fopenmp
 ```
- To build the library, navigate to the libgmm directory using the terminal and type

 ```
 make
 ```
- To build the MATLAB wrapper, run matlab/make.m from the MATLAB console.
- To build and the Python wrapper, navigate to the libgmm/python directory using the terminal and type

 ```
 python setup.py install
 ```

### Usage
- Using C API

 Refer to test.c
- MATLAB wrapper

 ```
 gmm = trainGMM(X, k, 'Name', 'Value', ...);
 ```
 Where,<br>
 X = NxD data matrix containing N data points, each of length D<br>
 k = Number of GMM components
 
 Optional name-value pairs:
 - CovType = Covariance matrix type: "diagonal" or "spherical". (Default "diagonal")
 - MaxIter = Maximum number of EM iterations. (Default 1000)
 - ConvergenceTol = Convergence tolerance. (Default 1e-6)
 - RegularizationValue = Regularization Value (small value added to covariance matrix to prevent it from being singular). (Default 1e-6)
 - InitMethod = GMM parameter initialization method. Can be 'random' or 'kmeans'. (Default 'random')
- Python wrapper

 ```
 import gmm
 gmm1 = gmm.GMM(k=1, CovType='diagonal', MaxIter=1000, ConvergenceTol=1e-6, RegularizationValue=1e-6, InitMethod='random')
 gmm1.fit(X)
 ```
 Where,<br>
 - X = NxD numpy matrix containing N data points, each of length D
 - k = Number of GMM components. (Default 1)
 - CovType = Covariance matrix type: "diagonal" or "spherical". (Default "diagonal")
 - MaxIter = Maximum number of EM iterations. (Default 1000)
 - ConvergenceTol = Convergence tolerance. (Default 1e-6)
 - RegularizationValue = Regularization Value (small value added to covariance matrix to prevent it from being singular). (Default 1e-6)
 - InitMethod = GMM parameter initialization method. Can be 'random' or 'kmeans'. (Default 'random')
