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

SGMM* sgmm_new(int M, int D)
{
	SGMM *gmm = (SGMM *) malloc(sizeof(SGMM));
	gmm->M = M;
	gmm->D = D;
	gmm->num_max_iter = 1000;
	gmm->converged = 0;
	gmm->tol = 0.000001;
	gmm->reg = 0.001;
	gmm->weights = (double *) malloc(gmm->M*sizeof(double));
	gmm->means = (double **) malloc(gmm->M*sizeof(double *));
	for (int k=0; k<gmm->M; k++)
		gmm->means[k] = (double *) malloc(gmm->D*sizeof(double));
	gmm->vars = (double *) malloc(gmm->M*sizeof(double));
	return gmm;
}

void sgmm_set_max_iter(SGMM *gmm, int num_max_iter)
{
	gmm->num_max_iter = num_max_iter;
}

void sgmm_set_convergence_tol(SGMM *gmm, double tol)
{
	gmm->tol = tol;
}

void sgmm_set_regularization_value(SGMM *gmm, double reg)
{
	gmm->reg = reg;
}

void sgmm_set_initialization_method(SGMM *gmm, const char *method)
{
	strcpy(gmm->init_method, method);
}

void sgmm_fit(SGMM *gmm, const double * const *X, int N)
{
	// Initialize SGMM parameters
	_sgmm_init_params(gmm, X, N);

	// Allocate memory for storing membership probabilities P(k | x_t)
	double **P_k_giv_xt = (double **) malloc(gmm->M*sizeof(double *));
	for (int k = 0; k < gmm->M; k++)
		P_k_giv_xt[k] = (double *) malloc(N*sizeof(double));

	// EM iterations
	double llh = 0, llh_prev = 0;
	for (int i_iter = 0; i_iter < gmm->num_max_iter; i_iter++)
	{
		// Perform one EM step
		llh_prev = llh;
		llh = _sgmm_em_step(gmm, X, N, P_k_giv_xt);
		//if (i_iter%20 == 0)
		IPrintf("Iter = %d, LLH = %lf\n", i_iter+1, llh);

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
		free(P_k_giv_xt[k]);
	free(P_k_giv_xt);
}

// TODO: Other initialization methods
void _sgmm_init_params(SGMM *gmm, const double * const *X, int N)
{
	if (strcmp(gmm->init_method, "random") == 0)
	{
		// Random initialization
		_sgmm_init_params_random(gmm, X, N);
	}
	else if (strcmp(gmm->init_method, "kmeans") == 0)
	{
		// K-means initialization
		_sgmm_init_params_kmeans(gmm, X, N);
	}
	else
	{
		// Default is random initialization
		_sgmm_init_params_random(gmm, X, N);
	}
}

// TODO: Unique sampling of data points for initializing component means
void _sgmm_init_params_random(SGMM *gmm, const double * const *X, int N)
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
		gmm->weights[k] += 1.0/gmm->M;
	
	// Initialize component variances to data variance
	double *mean = (double *) malloc(gmm->D*sizeof(double));
	for (int t=0; t<N; t++)
		_sgmm_vec_add(mean, X[t], 1, 1, gmm->D);
	_sgmm_vec_divide_by_scalar(mean, N, gmm->D);
	double var = 0;
	for (int t=0; t<N; t++)
		var += pow(_sgmm_vec_l2_dist(X[t], mean, gmm->D), 2);
	var = var/(N*gmm->D);
	for (int k=0; k<gmm->M; k++)
		gmm->vars[k] = var;

	// Fre memory used for storing mean
	free(mean);
}

// TODO: Unique sampling of data points for initializing component means
// TODO: Make K-means more efficient
void _sgmm_init_params_kmeans(SGMM *gmm, const double * const *X, int N)
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
	int *associations = (int *) malloc(N*sizeof(int));
	for (int i_iter = 0; i_iter < num_iter; i_iter++)
	{
		IPrintf(".");

		// Find assiciation of each data point
		for (int t = 0; t < N; t++)
		{
			double min_dist = _sgmm_vec_l2_dist(X[t], gmm->means[0], gmm->D);
			associations[t] = 0;
			for (int k=1; k<gmm->M; k++)
			{
				double dist = _sgmm_vec_l2_dist(X[t], gmm->means[k], gmm->D);
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
					_sgmm_vec_add(gmm->means[k], X[t], 1, 1, gmm->D);
				}
			}
			_sgmm_vec_divide_by_scalar(gmm->means[k], nk, gmm->D);
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
		gmm->vars[k] = 0;
		for (int t=0; t<N; t++)
		{
			if (associations[t] == k)
			{
				nk++;
				gmm->vars[k] += pow(_sgmm_vec_l2_dist(X[t], gmm->means[k], gmm->D), 2);
			}
		}
		gmm->vars[k] = gmm->vars[k]/(nk*gmm->D);
		if (gmm->vars[k] < gmm->reg)
			gmm->vars[k] = gmm->reg;
	}

	// Fre memory used for storing associations
	free(associations);
}

double _sgmm_em_step(SGMM *gmm, const double * const *X, int N, double **P_k_giv_xt)
{
	double llh;

	/* ---------------------------------------------- Expectation step */
	
	// Compute membership probabilities
	llh = _sgmm_compute_membership_prob(gmm, X, N, P_k_giv_xt);

	/* --------------------------------------------- Maximization step */

	// Update GMM parameters
	_sgmm_update_params(gmm, X, N, P_k_giv_xt);

	return llh;
}

double _sgmm_compute_membership_prob(SGMM *gmm, const double * const *X, int N, double **P)
{
	double llh = 0;

	// Populate the matrix with log(P(k | xt, gmm))
	#pragma omp parallel for reduction(+:llh)
	for (int t = 0; t < N; t++)
	{
		double max = -1;
		for (int k = 0; k < gmm->M; k++)
		{
			P[k][t] = log(gmm->weights[k]) + _sgmm_log_gaussian_pdf(X[t], gmm->means[k], gmm->vars[k], gmm->D);
			if (P[k][t] > max)
				max = P[k][t];
		}

		double llh_t = 0;
		for (int k=0; k<gmm->M; k++)
			llh_t += exp(P[k][t] - max);
		llh_t = max + log(llh_t);

		for (int k = 0; k < gmm->M; k++)
		{
			P[k][t] = exp(P[k][t] - llh_t);
		}

		llh += llh_t/N;
	}

	return llh;
}

void _sgmm_update_params(SGMM *gmm, const double * const *X, int N, double **P)
{
	#pragma omp parallel for
	for (int k=0; k<gmm->M; k++)
	{
		double sum_P_k = 0;
		double sum_xxP_k = 0;
		memset(gmm->means[k], 0, gmm->D*sizeof(int));
		for (int t=0; t<N; t++)
		{
			sum_P_k += P[k][t];
			sum_xxP_k += _sgmm_vec_dot_prod(X[t], X[t], gmm->D) * P[k][t];
			_sgmm_vec_add(gmm->means[k], X[t], 1, P[k][t], gmm->D);
		}
		_sgmm_vec_divide_by_scalar(gmm->means[k], sum_P_k, gmm->D);
		gmm->weights[k] = sum_P_k/N;
		gmm->vars[k] = (sum_xxP_k/sum_P_k - _sgmm_vec_dot_prod(gmm->means[k], gmm->means[k], gmm->D))/gmm->D;
		if (gmm->vars[k] < gmm->reg)
			gmm->vars[k] = gmm->reg;
	}
}

double _sgmm_log_gaussian_pdf(const double *x, const double *mean, double var, int D)
{
	double tmp = _sgmm_vec_l2_dist(x, mean, D);
	return -0.5 * log(2*PI) - 0.5 * D * log(var) - tmp*tmp/(2*var);
}

double _sgmm_vec_l2_dist(const double *x, const double *y, int D)
{
	double l2_dist_sq = 0;
	for (int i=0; i<D; i++)
	{
		double tmp = x[i] - y[i];
		l2_dist_sq += tmp*tmp;
	}
	return(sqrt(l2_dist_sq));
}

void _sgmm_vec_add(double *x, const double *y, double a, double b, int D)
{
	for (int i=0; i<D; i++)
		x[i] = a*x[i] + b*y[i];
}

void _sgmm_vec_divide_by_scalar(double *x, double a, int D)
{
	for (int i=0; i<D; i++)
		x[i] = x[i]/a;
}

double _sgmm_vec_dot_prod(const double *x, const double *y, int D)
{
	double prod = 0;
	for (int i=0; i<D; i++)
		prod += x[i]*y[i];
	return prod;
}

void sgmm_print_params(const SGMM *gmm)
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
		IPrintf("Var: %lf\n", gmm->vars[k]);
		IPrintf("\n");
	}
}

void sgmm_free(SGMM *gmm)
{
	free(gmm->weights);
	for (int k=0; k<gmm->M; k++)
		free(gmm->means[k]);
	free(gmm->means);
	free(gmm->vars);
	free(gmm);
}

