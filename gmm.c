/*
 *	gmm.c
 *	
 *	Contains definitions of functions for training
 *	Gaussian Mixture Models
 *
 *	Copyright (C) 2015 Sai Nitish Satyavolu
 */

#include "gmm.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
 #include <omp.h>
#endif

#ifdef MEX_COMPILE
 #include "mex.h"
 #define IPrintf mexPrintf
#else
 #define IPrintf printf
#endif

#define PI 3.14159265359

GMM* gmm_new(int M, int D, const char *cov_type)
{
	GMM *gmm = malloc(sizeof(GMM));

	// Set GMM settings
	gmm->M = M;
	gmm->D = D;
	gmm->num_max_iter = 1000;
	gmm->converged = 0;
	gmm->tol = 0.000001;
	gmm->reg = 0.001;
	gmm->init_method = RANDOM;
	if (strcmp(cov_type, "diagonal") == 0)
		gmm->cov_type = DIAGONAL;
	else if (strcmp(cov_type, "spherical") == 0)
		gmm->cov_type = SPHERICAL;
	else
	{
		IPrintf("WARNING: Invalid cov_type specified. Defaulting to DIAGONAL.\n");
		gmm->cov_type = DIAGONAL;
	}

	// Allocate memory for GMM parameters
	gmm->weights = malloc(gmm->M*sizeof(double));
	gmm->means = malloc(gmm->M*sizeof(double *));
	gmm->covars = malloc(gmm->M*sizeof(double *));
	for (int k=0; k<gmm->M; k++)
	{
		gmm->means[k] = malloc(gmm->D*sizeof(double));
		if (gmm->cov_type == DIAGONAL)
			gmm->covars[k] = malloc(gmm->D*sizeof(double));
		else if (gmm->cov_type == SPHERICAL)
			gmm->covars[k] = malloc(1*sizeof(double));
	}

	return gmm;
}

void gmm_set_max_iter(GMM *gmm, int num_max_iter)
{
	gmm->num_max_iter = num_max_iter;
}

void gmm_set_convergence_tol(GMM *gmm, double tol)
{
	gmm->tol = tol;
}

void gmm_set_regularization_value(GMM *gmm, double reg)
{
	gmm->reg = reg;
}

void gmm_set_initialization_method(GMM *gmm, const char *method)
{
	if (strcmp(method, "random") == 0)
		gmm->init_method = RANDOM;
	else if (strcmp(method, "kmeans") == 0)
		gmm->init_method = KMEANS;
	else
	{
		IPrintf("WARNING: Invalid init_method specified. Defaulting to RANDOM.\n");
		gmm->init_method = RANDOM;
	}
}

void gmm_fit(GMM *gmm, const double * const *X, int N)
{
	// Initialize GMM parameters
	_gmm_init_params(gmm, X, N);

	// Allocate memory for storing membership probabilities P(k | x_t)
	gmm->P_k_giv_xt = malloc(gmm->M*sizeof(double *));
	for (int k = 0; k < gmm->M; k++)
		gmm->P_k_giv_xt[k] = malloc(N*sizeof(double));

	// EM iterations
	double llh = 0, llh_prev = 0;
	for (int i_iter = 0; i_iter < gmm->num_max_iter; i_iter++)
	{
		// Perform one EM step
		llh_prev = llh;
		llh = _gmm_em_step(gmm, X, N);
		//if (i_iter%20 == 0)
		IPrintf("Iter = %d, LLH = %lf\n", i_iter+1, llh);
		if ((llh > 0) == 0 && (llh <= 0) == 0)
		{
			IPrintf("WARNING: Encountered NaN value at iteration: %d\n", i_iter+1);
			gmm_print_params(gmm);
			break;
		}
		
		// Check for convergence
		if (i_iter > 2 && fabs((llh - llh_prev)/llh_prev) < gmm->tol)
		{
			gmm->converged = 1;
			IPrintf("EM algorithm converged after %d iterations.\n", i_iter+1);
			break;
		}
	}

	// Free memory used for storing membership probabilities
	for (int k = 0; k < gmm->M; k++)
		free(gmm->P_k_giv_xt[k]);
	free(gmm->P_k_giv_xt);
}

// TODO: Other initialization methods
void _gmm_init_params(GMM *gmm, const double * const *X, int N)
{
	if (gmm->init_method == RANDOM)
	{
		// Random initialization
		_gmm_init_params_random(gmm, X, N);
	}
	else if (gmm->init_method == KMEANS)
	{
		// K-means initialization
		_gmm_init_params_kmeans(gmm, X, N);
	}
	else
	{
		// Default is random initialization
		_gmm_init_params_random(gmm, X, N);
	}
}

// TODO: Unique sampling of data points for initializing component means
void _gmm_init_params_random(GMM *gmm, const double * const *X, int N)
{
	// Initialize means to randomly chosen samples
	srand(time(NULL));
	for (int k=0; k<gmm->M; k++)
	{
		int r = rand()%N;
		memcpy(gmm->means[k], X[r], gmm->D*sizeof(double));
	}

	// Initialize component weights to same value
	for (int k=0; k<gmm->M; k++)
		gmm->weights[k] = 1.0/gmm->M;
	
	// Initialize component variances to data variance
	double *mean = calloc(gmm->D, sizeof(double));
	for (int t=0; t<N; t++)
		_gmm_vec_add(mean, X[t], 1, 1, gmm->D);
	_gmm_vec_divide_by_scalar(mean, N, gmm->D);
	if (gmm->cov_type == DIAGONAL)
	{
		double *vars = malloc(gmm->D*sizeof(double));
		for (int i=0; i<gmm->D; i++)
		{
			vars[i] = 0;
			for (int t=0; t<N; t++)
				vars[i] += _gmm_pow2(X[t][i] - mean[i]);
			vars[i] = vars[i]/N;
		}
		for (int k=0; k<gmm->M; k++)
			memcpy(gmm->covars[k], vars, gmm->D*sizeof(double));
		free(vars);
	}
	else if (gmm->cov_type == SPHERICAL)
	{
		double var = 0;
		for (int t=0; t<N; t++)
			var += _gmm_pow2(_gmm_vec_l2_dist(X[t], mean, gmm->D));
		var = var/(N*gmm->D);
		for (int k=0; k<gmm->M; k++)
			gmm->covars[k][0] = var;	
	}

	// Fre memory used for storing mean
	free(mean);
}

// TODO: Handle empty clusters in K-means
// TODO: Unique sampling of data points for initializing component means
// TODO: Make K-means more efficient
void _gmm_init_params_kmeans(GMM *gmm, const double * const *X, int N)
{
	const int num_iter = 10;

	// Initialize means to randomly chosen samples
	srand(time(NULL));
	for (int k=0; k<gmm->M; k++)
	{
		int r = rand()%N;
		memcpy(gmm->means[k], X[r], gmm->D*sizeof(double));
	}

	// K-means iterative algorithm
	int *associations = malloc(N*sizeof(int));
	for (int i_iter = 0; i_iter < num_iter; i_iter++)
	{
		IPrintf(".");

		// Find assiciation of each data point
		for (int t = 0; t < N; t++)
		{
			double min_dist = _gmm_vec_l2_dist(X[t], gmm->means[0], gmm->D);
			associations[t] = 0;
			for (int k=1; k<gmm->M; k++)
			{
				double dist = _gmm_vec_l2_dist(X[t], gmm->means[k], gmm->D);
				if (dist < min_dist)
				{
					min_dist = dist;
					associations[t] = k;
				}
			}
		}

		// Update mean of each cluster
		for (int k=0; k<gmm->M; k++)
		{
			memset(gmm->means[k], 0, gmm->D*sizeof(double));
			int nk = 0;
			for (int t=0; t<N; t++)
			{
				if (associations[t] == k)
				{
					nk++;
					_gmm_vec_add(gmm->means[k], X[t], 1, 1, gmm->D);
				}
			}
			_gmm_vec_divide_by_scalar(gmm->means[k], nk, gmm->D);
		}
	}
	IPrintf("\n");

	// Initialize component weights to fraction of associations
	memset(gmm->weights, 0, gmm->M*sizeof(double));
	for (int t=0; t<N; t++)
		gmm->weights[associations[t]] += 1.0/N;
	
	// Initialize component variances to variances in each cluster
	for (int k=0; k<gmm->M; k++)
	{
		int nk = 0;
		if (gmm->cov_type == SPHERICAL)
			gmm->covars[k][0] = 0;
		else if (gmm->cov_type == DIAGONAL)
			memset(gmm->covars[k], 0, gmm->D*sizeof(double));
		for (int t=0; t<N; t++)
		{
			if (associations[t] == k)
			{
				nk++;
				if (gmm->cov_type == SPHERICAL)
					gmm->covars[k][0] += _gmm_pow2(_gmm_vec_l2_dist(X[t], gmm->means[k], gmm->D));
				else if (gmm->cov_type == DIAGONAL)
				{
					for (int i=0; i<gmm->D; i++)
						gmm->covars[k][i] += _gmm_pow2(X[t][i] - gmm->means[k][i]);
				}
			}
		}
		if (gmm->cov_type == SPHERICAL)
		{	
			gmm->covars[k][0] = gmm->covars[k][0]/(nk*gmm->D);
			if (gmm->covars[k][0] < gmm->reg)
				gmm->covars[k][0] = gmm->reg;
		}
		else if (gmm->cov_type == DIAGONAL)
		{
			_gmm_vec_divide_by_scalar(gmm->covars[k], nk, gmm->D);
			for (int i=0; i<gmm->D; i++)
			{
				if (gmm->covars[k][i] < gmm->reg)
					gmm->covars[k][i] = gmm->reg;
			}
		}
	}

	// Fre memory used for storing associations
	free(associations);
}

double _gmm_em_step(GMM *gmm, const double * const *X, int N)
{
	double llh;

	/* ---------------------------------------------- Expectation step */
	
	// Compute membership probabilities
	llh = _gmm_compute_membership_prob(gmm, X, N);

	/* --------------------------------------------- Maximization step */

	// Update GMM parameters
	_gmm_update_params(gmm, X, N);

	return llh;
}

double _gmm_compute_membership_prob(GMM *gmm, const double * const *X, int N)
{
	double llh = 0;

	// Populate the matrix with log(P(k | xt, gmm))
	#pragma omp parallel for reduction(+:llh)
	for (int t = 0; t < N; t++)
	{
		double max = -1;
		for (int k = 0; k < gmm->M; k++)
		{
			gmm->P_k_giv_xt[k][t] = log(gmm->weights[k]) + _gmm_log_gaussian_pdf(X[t], gmm->means[k], gmm->covars[k], gmm->D, gmm->cov_type);
			if (gmm->P_k_giv_xt[k][t] > max)
				max = gmm->P_k_giv_xt[k][t];
		}

		double llh_t = 0;
		for (int k=0; k<gmm->M; k++)
			llh_t += exp(gmm->P_k_giv_xt[k][t] - max);
		llh_t = max + log(llh_t);

		for (int k = 0; k < gmm->M; k++)
		{
			gmm->P_k_giv_xt[k][t] = exp(gmm->P_k_giv_xt[k][t] - llh_t);
		}

		llh += llh_t/N;
	}

	return llh;
}

void _gmm_update_params(GMM *gmm, const double * const *X, int N)
{
	if (gmm->cov_type == SPHERICAL)
	{
		#pragma omp parallel for
		for (int k=0; k<gmm->M; k++)
		{
			double sum_P_k = 0;
			double sum_xxP_k = 0;
			memset(gmm->means[k], 0, gmm->D*sizeof(double));
			for (int t=0; t<N; t++)
			{
				sum_P_k += gmm->P_k_giv_xt[k][t];
				sum_xxP_k += _gmm_vec_dot_prod(X[t], X[t], gmm->D) * gmm->P_k_giv_xt[k][t];
				_gmm_vec_add(gmm->means[k], X[t], 1, gmm->P_k_giv_xt[k][t], gmm->D);
			}
			_gmm_vec_divide_by_scalar(gmm->means[k], sum_P_k, gmm->D);
			gmm->weights[k] = sum_P_k/N;
			gmm->covars[k][0] = (sum_xxP_k/sum_P_k - _gmm_vec_dot_prod(gmm->means[k], gmm->means[k], gmm->D))/gmm->D;
			if (gmm->covars[k][0] < gmm->reg)
				gmm->covars[k][0] = gmm->reg;
		}	
	}
	else if (gmm->cov_type == DIAGONAL)
	{
		#pragma omp parallel for
		for (int k=0; k<gmm->M; k++)
		{
			double sum_P_k = 0;
			memset(gmm->means[k], 0, gmm->D*sizeof(double));
			for (int t=0; t<N; t++)
			{
				sum_P_k += gmm->P_k_giv_xt[k][t];
				_gmm_vec_add(gmm->means[k], X[t], 1, gmm->P_k_giv_xt[k][t], gmm->D);
			}
			gmm->weights[k] = sum_P_k/N;
			_gmm_vec_divide_by_scalar(gmm->means[k], sum_P_k, gmm->D);
			
			memset(gmm->covars[k], 0, gmm->D*sizeof(double));
			for (int t=0; t<N; t++)
			{
				for (int i=0; i<gmm->D; i++)
					gmm->covars[k][i] += gmm->P_k_giv_xt[k][t]*_gmm_pow2(X[t][i] - gmm->means[k][i]);
			}
			_gmm_vec_divide_by_scalar(gmm->covars[k], sum_P_k, gmm->D);
			for (int i=0; i<gmm->D; i++)
			{
				if (gmm->covars[k][i] < gmm->reg)
					gmm->covars[k][i] = gmm->reg;
			}
		}
	}
}

// TODO: Pre-compute det of covariance matrix
double _gmm_log_gaussian_pdf(const double *x, const double *mean, const double *covar, int D, CovType cov_type)
{
	double result = 0;
	if (cov_type == SPHERICAL)
		result = -0.5 * D * log(2*PI*covar[0]) - _gmm_pow2(_gmm_vec_l2_dist(x, mean, D))/(2*covar[0]);
	else if (cov_type == DIAGONAL)
	{
		double det = 1;
		for (int i=0; i<D; ++i)
			det *= covar[i];

		result = -0.5 * D * log(2*PI) - 0.5 * log(det);
		for (int i=0; i<D; ++i)
			result -= _gmm_pow2(x[i] - mean[i])/(2*covar[i]);
	}
	return result;
}

double _gmm_vec_l2_dist(const double *x, const double *y, int D)
{
	double l2_dist_sq = 0;
	for (int i=0; i<D; i++)
	{
		l2_dist_sq += _gmm_pow2(x[i] - y[i]);
	}
	return(sqrt(l2_dist_sq));
}

void _gmm_vec_add(double *x, const double *y, double a, double b, int D)
{
	for (int i=0; i<D; i++)
		x[i] = a*x[i] + b*y[i];
}

void _gmm_vec_divide_by_scalar(double *x, double a, int D)
{
	for (int i=0; i<D; i++)
		x[i] = x[i]/a;
}

double _gmm_vec_dot_prod(const double *x, const double *y, int D)
{
	double prod = 0;
	for (int i=0; i<D; i++)
		prod += x[i]*y[i];
	return prod;
}

double _gmm_pow2(double x)
{
	return x*x;
}

void gmm_print_params(const GMM *gmm)
{
	for (int k=0; k<gmm->M; k++)
	{
		IPrintf("Component: %d\n", k+1);
		IPrintf("Weight: %lf\n", gmm->weights[k]);
		if (gmm->D < 50)
		{
			IPrintf("Mean: ");
			for (int i=0; i<gmm->D; i++)
				IPrintf("%lf, ", gmm->means[k][i]);
			IPrintf("\n");
		}
		if (gmm->cov_type == SPHERICAL)
			IPrintf("Var: %lf\n", gmm->covars[k][0]);
		else if (gmm->cov_type == DIAGONAL)
		{
			IPrintf("Var: ");
			int i=0;
			for (; i<5 && i<gmm->D; i++)
				IPrintf("%lf, ", gmm->covars[k][i]);
			if (i < gmm->D)
				IPrintf("...");
			IPrintf("\n");
		}
		IPrintf("\n");
	}
}

void gmm_free(GMM *gmm)
{
	free(gmm->weights);
	for (int k=0; k<gmm->M; k++)
	{	
		free(gmm->means[k]);
		free(gmm->covars[k]);
	}
	free(gmm->means);
	free(gmm->covars);
	free(gmm);
}

