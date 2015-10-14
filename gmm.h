/*
 *	gmm.h
 *	
 *	Contains declarations of functions for training
 *	Gaussian Mixture Models
 *
 *	Copyright (C) 2015 Sai Nitish Satyavolu
 */

#ifndef GMM_H
#define GMM_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 *	Type for storing GMM parameter initialization method	
 */
typedef enum {RANDOM, KMEANS} InitMethod;

/*
 *	Type for storing the type of GMM covariance matrix
 */
typedef enum {DIAGONAL, SPHERICAL} CovType;

/*
 *	The GMM structure
 */
typedef struct _GMM
{
	/* --------------------------------- Settings */
	int M;						// Number of components
	int D;						// Number of features
	int num_max_iter;			// Maximum number of iterations
	int converged;				// Convergence status
	double tol;					// Convergence tolerance
	double reg;					// Regularization value
	InitMethod init_method;		// Initialization method
	CovType cov_type;			// Covariance type
	/* --------------------------- GMM Parameters */
	double *weights;			// Component weights
	double **means; 			// Component means
	double **covars;			// Component variances
	/* ---------------------- Auxiliary variables */
	double **P_k_giv_xt;		// Membership probability matrix
} GMM;

/*
 *	Function for initializing a new GMM
 */
GMM* gmm_new(int M, int D, const char *cov_type);

/*
 *	Function to set maximum number of EM iterations
 */
void gmm_set_max_iter(GMM *gmm, int num_max_iter);

/*
 *	Function to set EM convergence tolerance
 */
void gmm_set_convergence_tol(GMM *gmm, double tol);

/*
 *	Function to set regularization value of covariance matrix
 */
void gmm_set_regularization_value(GMM *gmm, double reg);

/*
 *	Function to set GMM parameter initialization method
 */
void gmm_set_initialization_method(GMM *gmm, const char *method);

/*
 *	Function to fit a GMM on a given set of data points
 */
void gmm_fit(GMM *gmm, const double * const *X, int N);

/*
 *	Function to print the GMM parameters
 */
void gmm_print_params(const GMM *gmm);

/*
 *	Function to free the GMM
 */
void gmm_free(GMM *gmm);

/*
 *	Internal functions (do not call them!)
 */
void _gmm_init_params(GMM *gmm, const double * const *X, int N);
void _gmm_init_params_random(GMM *gmm, const double * const *X, int N);
void _gmm_init_params_kmeans(GMM *gmm, const double * const *X, int N);
double _gmm_em_step(GMM *gmm, const double * const *X, int N);
double _gmm_compute_membership_prob(GMM *gmm, const double * const *X, int N);
void _gmm_update_params(GMM *gmm, const double * const *X, int N);
double _gmm_log_gaussian_pdf(const double *x, const double *mean, const double *covar, int D, CovType cov_type);
double _gmm_vec_l2_dist(const double *x, const double *y, int D);
void _gmm_vec_add(double *x, const double *y, double a, double b, int D);
void _gmm_vec_divide_by_scalar(double *x, double a, int D);
double _gmm_vec_dot_prod(const double *x, const double *y, int D);
double _gmm_pow2(double x);


#ifdef __cplusplus
}
#endif

#endif
