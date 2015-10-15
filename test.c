 #define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "gmm.h"

int main()
{
	const char *data_filenm = "sample_data.txt";
	const int gmm_num_components = 3;

	// Load data from file
	FILE *fp = fopen(data_filenm, "r");
	if (fp == NULL)
	{
		printf("ERROR: File 'sample_data.txt' not found.\nRun scripts/generate_data.py to generate sample data.\n");
		exit(1);
	}
	int N = 0, D = 0;
	size_t bytes_read, len = 0;
	char *line = NULL;
	while ((bytes_read = getline(&line, &len, fp)) != -1)
	{
		if (bytes_read > 0)
			N++;
	}
	rewind(fp);
	len = 0;
	getline(&line, &len, fp);
	char *token = strtok(line, " \n");
	while (token != NULL)
	{
		D++;
		token = strtok(NULL, " \n");
	}
	double **X = (double **) malloc(N*sizeof(double *));
	for (int t=0; t<N; t++)
		X[t] = (double *) malloc(D*sizeof(double));
	rewind(fp);
	line = NULL; len = 0;
	for (int t=0; t<N; t++)
	{
		getline(&line, &len, fp);
		token = strtok(line, " \n");
		X[t][0] = atof(token);
		for (int i=1; i<D; i++)
		{
			token = strtok(NULL, " \n");
			X[t][i] = atof(token);
		}
	}
	fclose(fp);

	// Train the SGMM
	GMM *gmm = gmm_new(gmm_num_components, D, "diagonal");
	gmm_set_convergence_tol(gmm, 1e-6);
	gmm_set_regularization_value(gmm, 1e-6);
	gmm_set_initialization_method(gmm, "random");
	struct timeval st, en;
	gettimeofday(&st, NULL);
	gmm_fit(gmm, X, N);
	gettimeofday(&en, NULL);
	printf("Time elapsed = %lf s\n", (en.tv_sec-st.tv_sec) + (1e-6)*(en.tv_usec-st.tv_usec));
	gmm_print_params(gmm);
	gmm_free(gmm);

	// Free data
	for (int t=0; t<N; t++)
		free(X[t]);
	free(X);

	return 0;
}
