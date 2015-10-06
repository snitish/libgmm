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

typedef struct _SGMM
{
	/* ---------------------------- Settings */
	int M;					// Number of components
	int D;					// Number of features
	int num_max_iter;		// Maximum number of iterations
	int converged;			// Convergence status
	double tol;				// Convergence tolerance
	double reg;				// Regularization value
	char init_method[20];	// Initialization method
	/* ---------------------- GMM Parameters */
	double *weights;		// Component weights
	double **means; 		// Component means
	double *vars;			// Component variances
} SGMM;
SGMM* sgmm_new(int M, int D);
void sgmm_set_max_iter(SGMM *gmm, int num_max_iter);
void sgmm_set_convergence_tol(SGMM *gmm, double tol);
void sgmm_set_regularization_value(SGMM *gmm, double reg);
void sgmm_set_initialization_method(SGMM *gmm, const char *method);
void sgmm_fit(SGMM *gmm, const double * const *X, int N);
void _sgmm_init_params(SGMM *gmm, const double * const *X, int N);
void _sgmm_init_params_random(SGMM *gmm, const double * const *X, int N);
void _sgmm_init_params_kmeans(SGMM *gmm, const double * const *X, int N);
double _sgmm_em_step(SGMM *gmm, const double * const *X, int N, double **P_k_giv_xt);
double _sgmm_compute_membership_prob(SGMM *gmm, const double * const *X, int N, double **P_k_giv_xt);
void _sgmm_update_params(SGMM *gmm, const double * const *X, int N, double **P);
double _sgmm_log_gaussian_pdf(const double *x, const double *mean, double var, int D);
double _sgmm_vec_l2_dist(const double *x, const double *y, int D);
void _sgmm_vec_add(double *x, const double *y, double a, double b, int D);
void _sgmm_vec_divide_by_scalar(double *x, double a, int D);
double _sgmm_vec_dot_prod(const double *x, const double *y, int D);
void sgmm_print_params(const SGMM *gmm);
void sgmm_free(SGMM *gmm);

#ifdef __cplusplus
}
#endif

#endif
