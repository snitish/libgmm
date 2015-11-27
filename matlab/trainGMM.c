/*
 *	trainGMM.c
 *
 *	MEX wrapper for libgmm
 *
 *	Usage:
 *		[gmm] = trainGMM(X, k, varargin)
 */

#include "mex.h"
#include "../gmm.h"
#include <stdlib.h>
#include <string.h>

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
	/* -------------------------------------------------- Validate input arguments */
	
	// Make sure there are at least two input arguments
	if (nrhs < 2)
	{
		mexErrMsgIdAndTxt("trainGMM:nrhs", "At least two inputs required.");
	}

	// Make sure the first argument is a double matrix
	if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxGetNumberOfDimensions(prhs[0]) > 2)
	{
		mexErrMsgIdAndTxt("trainGMM:invalidArgument", "Training data (X) must be a 2D matrix of doubles.");
	}

	// Make sure the second argument is a scalar
	if( !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1)
	{
		mexErrMsgIdAndTxt("trainGMM:invalidArgument", "Number of components (k) must be a scalar.");
	}

	/* ------------------------------------------------- Validate output arguments */

	// Make sure there is at most one output argument
	if (nlhs > 1)
	{
		mexErrMsgIdAndTxt("trainGMM:nlhs", "The function returns only one output variable.");
	}

	/* ------------------------------------------------------- Get input arguments */

	double *X = mxGetPr(prhs[0]);
	int M = (int) (mxGetScalar(prhs[1]) + 0.5);
	int N = mxGetM(prhs[0]);
	int D = mxGetN(prhs[0]);

	// Get optional arguments
	double tol = 0.000001;
	double reg = 0.000001;
	int max_iter = 1000;
	char init_method[20]; strcpy(init_method, "random");
	char cov_type[20]; strcpy(cov_type, "diagonal");
	char error_msg[200];
	if (nrhs > 2)
	{
		int i_rhs = 2;
		while (i_rhs < nrhs)
		{
			// Make sure prhs[i_rhs] is a string
			if (!mxIsChar(prhs[i_rhs]))
			{
				mexErrMsgIdAndTxt("trainGMM:invalidArgument", "Optional argument name must be a string.");
			}

			// Get argument name
			char arg_name[50];
			mxGetString(prhs[i_rhs], arg_name, (mwSize) 50);

			// Make sure value argument exists
			if (i_rhs + 1 >= nrhs)
			{
				sprintf(error_msg, "Expecting value for parameter '%s'", arg_name);
				mexErrMsgIdAndTxt("trainGMM:argumentMissing", error_msg);
			}

			// Parse argument name and get value
			if (strcmp(arg_name, "RegularizationValue") == 0)
			{
				// Make sure argument value is a scalar
				if( !mxIsDouble(prhs[i_rhs + 1]) || mxIsComplex(prhs[i_rhs + 1]) || mxGetNumberOfElements(prhs[i_rhs + 1]) != 1)
				{
					mexErrMsgIdAndTxt("trainGMM:invalidArgument", "RegularizationValue must be a scalar.");
				}
				reg = mxGetScalar(prhs[i_rhs+1]);
			}
			else if (strcmp(arg_name, "ConvergenceTol") == 0)
			{
				// Make sure argument value is a scalar
				if( !mxIsDouble(prhs[i_rhs + 1]) || mxIsComplex(prhs[i_rhs + 1]) || mxGetNumberOfElements(prhs[i_rhs + 1]) != 1)
				{
					mexErrMsgIdAndTxt("trainGMM:invalidArgument", "ConvergenceTol must be a scalar.");
				}
				tol = mxGetScalar(prhs[i_rhs+1]);
			}
			else if (strcmp(arg_name, "MaxIter") == 0)
			{
				// Make sure argument value is a scalar
				if( !mxIsDouble(prhs[i_rhs + 1]) || mxIsComplex(prhs[i_rhs + 1]) || mxGetNumberOfElements(prhs[i_rhs + 1]) != 1)
				{
					mexErrMsgIdAndTxt("trainGMM:invalidArgument", "MaxIter must be a scalar.");
				}
				max_iter = mxGetScalar(prhs[i_rhs+1]);
			}
			else if (strcmp(arg_name, "InitMethod") == 0)
			{
				// Make sure argument value is a string
				if( !mxIsChar(prhs[i_rhs + 1]))
				{
					mexErrMsgIdAndTxt("trainGMM:invalidArgument", "InitMethod must be a string.");
				}
				mxGetString(prhs[i_rhs+1], init_method, (mwSize) 20);
			}
			else if (strcmp(arg_name, "CovType") == 0)
			{
				// Make sure argument value is a string
				if( !mxIsChar(prhs[i_rhs + 1]))
				{
					mexErrMsgIdAndTxt("trainGMM:invalidArgument", "CovType must be a string.");
				}
				mxGetString(prhs[i_rhs+1], cov_type, (mwSize) 20);
			}
			else
			{
				sprintf(error_msg, "Unknown parameter '%s'", arg_name);
				mexErrMsgIdAndTxt("trainGMM:invalidArgument", error_msg);
			}

			i_rhs += 2;
		}
	}
	
	/* ------------------------------------------------------------ Train the SGMM */	

	// Train the SGMM
	GMM *gmm = sgmm_new(M, D, cov_type);
	gmm_set_convergence_tol(gmm, tol);
	gmm_set_regularization_value(gmm, reg);
	gmm_set_max_iter(gmm, max_iter);
	gmm_set_initialization_method(gmm, init_method);
	gmm_fit(gmm, X, N);

	/* ------------------------------------------------ Build output GMM structure */

	//Initialize structure with three fields
	const mwSize dims[2] = {1, 1};
	const char *field_names[] = {"numComponents", "featureLength", "weights", "means", "covars"};
	plhs[0] = mxCreateStructArray(2, dims, 5, field_names);

    // Add "numComponents" field to structure
	mxArray *mx_numComponents = mxCreateDoubleScalar(M);
	mxSetField(plhs[0], 0, "numComponents", mx_numComponents);
    
    // Add "featureLength" field to structure
	mxArray *mx_featureLength = mxCreateDoubleScalar(D);
	mxSetField(plhs[0], 0, "featureLength", mx_featureLength);
    
	// Add "weights" field to structure
	mxArray *mx_weights = mxCreateDoubleMatrix(M, 1, mxREAL);
	double *p = mxGetPr(mx_weights);
	memcpy(p, gmm->weights, M*sizeof(double));
	mxSetField(plhs[0], 0, "weights", mx_weights);

	// Add "means" field to structure
	mxArray *mx_means = mxCreateDoubleMatrix(M, D, mxREAL);
	p = mxGetPr(mx_means);
	for (int k=0; k<M; k++)
		for (int i=0; i<D; i++)
			p[k+i*M] = gmm->means[k][i];
	mxSetField(plhs[0], 0, "means", mx_means);

	// Add "covars" field to structure
	mxArray *mx_covars = mxCreateDoubleMatrix(M, 1, mxREAL);
	p = mxGetPr(mx_covars);
	int cov_len = (gmm->cov_type == SPHERICAL)? 1 : D;
	for (int k=0; k<M; k++)
		for (int i=0; i<cov_len; i++)
			p[k+i*M] = gmm->covars[k][i];
	mxSetField(plhs[0], 0, "covars", mx_covars);

	/* ------------------------------------------------------------------- Cleanup */

	// Free gmm
	sgmm_free(gmm);
	
}

