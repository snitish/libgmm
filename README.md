# libgmm
A library for training Gaussian Mixture Models written in C.

### How to build
To build the library, navigate to the libgmm directory using the terminal and type
```
make
```
To also build the MATLAB wrapper, run matlab/make.m from the MATLAB console.

### Usage
- Using C API

 Refer to test.c
- MATLAB wrapper

 ```
 gmm = trainSphericalGMM(X, k, 'Name', 'Value', ...);
 ```
 Where,<br>
 X = NxD data matrix containing N data points, each of length D<br>
 k = Number of GMM components
 
 Optional name-value pairs:
 - MaxIter = Maximum number of EM iterations. (Default 1000)
 - ConvergenceTol = Convergence tolerance. (Default 1e-6)
 - RegularizationValue = Regularization Value (small value added to covariance matrix to prevent it from being singular). (Default 1e-6)
 - InitMethod = GMM parameter initialization method. Can be 'random' or 'kmeans'. (Default 'random')
