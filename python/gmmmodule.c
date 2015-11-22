#include <Python/Python.h>
#include <structmember.h>
#include <Numpy/arrayobject.h>
#include "../gmm.h"

typedef struct {
	PyObject_HEAD
	/* Type-specific fields go here. */
	PyArrayObject *weights;
	PyArrayObject *means;
	PyArrayObject *covars;
	int k;
	PyObject *cov_type;
	PyObject *init_method;
	double tol;
	int max_iter;
	double reg;
} GMMObject;

static int
GMM_init(GMMObject *self, PyObject *args, PyObject *keywds)
{
	// Parse the arguments
	static char *kwlist[] = {"k",
							 "CovType",
							 "InitMethod",
							 "ConvergenceTol",
							 "MaxIter",
							 "RegularizationValue",
							 NULL};
	int k = 1, max_iter = 1000;
	char *cov_type = "diagonal", *init_method = "random";
	double tol = 1e-6, reg = 1e-6;
	if (!PyArg_ParseTupleAndKeywords(args, 
									 keywds, 
									 "|issdid", 
									 kwlist,
									 &k, 
									 &cov_type,
									 &init_method,
									 &tol,
									 &max_iter,
									 &reg))
	{
		return -1;
	}

	// Fill GMM object
	self->k = k;
	self->cov_type = PyString_FromString(cov_type);
	self->init_method = PyString_FromString(init_method);
	self->tol = tol;
	self->max_iter = max_iter;
	self->reg = reg;

	return 0;
}

static void
GMM_dealloc(GMMObject* self)
{
	Py_XDECREF(self->weights);
	Py_XDECREF(self->means);
	Py_XDECREF(self->covars);
	self->ob_type->tp_free((PyObject *) self);
}

static PyObject *
GMM_fit(GMMObject *self, PyObject *args, PyObject *keywds)
{
	// Parse the arguments
	static char *kwlist[] = {"X", 
							 NULL};
	PyObject *X_obj;
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &X_obj))
	{
		return NULL;
	}

	// Get data matrix from numpy array
	PyArrayObject *X_array = (PyArrayObject *) PyArray_ContiguousFromObject(X_obj, PyArray_DOUBLE, 2, 2);
	if (X_array == NULL)
	{
		printf("Data matrix (X) in bad format.\n");
		return NULL;
	}
	if (PyArray_NDIM(X_array) != 2)
	{
		printf("Data matrix (X) must be a 2D matrix.\n");
		return NULL;
	}
	int N = (int) PyArray_DIM(X_array, 0);
	int D = (int) PyArray_DIM(X_array, 1);
	double **X = malloc(N*sizeof(double *));
	for (int t=0; t<N; t++)
		X[t] = (double *) X_array->data + D*t;

	// Train the GMM
	GMM *gmm = gmm_new(self->k, D, PyString_AsString(self->cov_type));
	gmm_set_convergence_tol(gmm, self->tol);
	gmm_set_regularization_value(gmm, self->reg);
	gmm_set_initialization_method(gmm, PyString_AsString(self->init_method));
	gmm_fit(gmm, X, N);
	
	// Copy component weights to GMMObject
	npy_intp weights_dims[1] = {self->k};
	self->weights = (PyArrayObject *) PyArray_SimpleNew(1, weights_dims, NPY_DOUBLE);
	memcpy((double *) self->weights->data, gmm->weights, self->k*sizeof(double));

	// Copy component means to GMMObject
	npy_intp means_dims[2] = {self->k, D};
	self->means = (PyArrayObject *) PyArray_SimpleNew(2, means_dims, NPY_DOUBLE);
	for (int ik=0; ik<self->k; ik++)
		memcpy((double *) self->means->data + ik*D, gmm->means[ik], D*sizeof(double));

	// Copy component covariances to GMMObject
	int covar_len = 0;
	if (gmm->cov_type == SPHERICAL)
		covar_len = 1;
	else if (gmm->cov_type == DIAGONAL)
		covar_len = D;
	npy_intp covars_dims[2] = {self->k, covar_len};
	self->covars = (PyArrayObject *) PyArray_SimpleNew(2, covars_dims, NPY_DOUBLE);
	for (int ik=0; ik<self->k; ik++)
		memcpy((double *) self->covars->data + ik*covar_len, gmm->covars[ik], covar_len*sizeof(double));

	// Free the GMM object
	gmm_free(gmm);

	// Free the data matrix and pyobjects
	free(X);
	Py_DECREF(X_array);

	return Py_BuildValue("");
}

static PyObject *
GMM_score(GMMObject *self, PyObject *args, PyObject *keywds)
{
	// Parse the arguments
	static char *kwlist[] = {"X", 
							 NULL};
	PyObject *X_obj;
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &X_obj))
	{
		return NULL;
	}

	// Get data matrix from numpy array
	PyArrayObject *X_array = (PyArrayObject *) PyArray_ContiguousFromObject(X_obj, PyArray_DOUBLE, 2, 2);
	if (X_array == NULL)
	{
		printf("Data matrix (X) in bad format.\n");
		return NULL;
	}
	if (PyArray_NDIM(X_array) != 2)
	{
		printf("Data matrix (X) must be a 2D matrix.\n");
		return NULL;
	}
	int N = (int) PyArray_DIM(X_array, 0);
	int D = (int) PyArray_DIM(X_array, 1);
	if (D != PyArray_DIM(self->means, 1))
	{
		printf("Invalid dimensions of data matrix X.\n");
		return NULL;
	}
	double **X = malloc(N*sizeof(double *));
	for (int t=0; t<N; t++)
		X[t] = (double *) X_array->data + D*t;

	// Initialize GMM from parameters
	GMM *gmm = malloc(sizeof(GMM));
	gmm->M = self->k;
	gmm->D = D;
	int covar_len = 0;
	const char *cov_type = PyString_AsString(self->cov_type);
	if (strcmp(cov_type, "spherical") == 0)
	{
		covar_len = 1;
		gmm->cov_type = SPHERICAL;
	}
	else if (strcmp(cov_type, "diagonal") == 0)
	{
		covar_len = D;
		gmm->cov_type = DIAGONAL;
	}
	double **means = malloc(self->k*sizeof(double *));
	double **covars = malloc(self->k*sizeof(double *));
	for (int k=0; k<self->k; k++)
	{
		means[k] = (double *) self->means->data + k*D;
		covars[k] = (double *) self->covars->data + k*covar_len;
	}
	gmm->weights = (double *) self->weights->data;
	gmm->means = means;
	gmm->covars = covars;
	
	// Score the data points
	double llh = gmm_score(gmm, X, N);

	// Free the GMM object
	free(gmm);

	// Free the parameter arrays
	free(means);
	free(covars);

	// Free the data matrix and pyobjects
	free(X);
	Py_DECREF(X_array);

	return Py_BuildValue("");
}

static PyMemberDef GMM_members[] = {
	{"weights", T_OBJECT, offsetof(GMMObject, weights), 0, "Component weights"},
	{"means", T_OBJECT, offsetof(GMMObject, means), 0, "Component means"},
	{"covars", T_OBJECT, offsetof(GMMObject, covars), 0, "Component covariances"},
	{"k", T_INT, offsetof(GMMObject, k), 0, "Number of components"},
	{"cov_type", T_OBJECT, offsetof(GMMObject, cov_type), 0, "Covariance matrix type"},
	{"init_method", T_OBJECT, offsetof(GMMObject, init_method), 0, "Parameter initialization method"},
	{"tol", T_DOUBLE, offsetof(GMMObject, tol), 0, "Convergence tolerance"},
	{"max_iter", T_INT, offsetof(GMMObject, max_iter), 0, "Maximum number of EM iterations"},
	{"reg", T_DOUBLE, offsetof(GMMObject, reg), 0, "Regularization value"},
	{NULL}  /* Sentinel */
};

static PyMethodDef GMM_methods[] = {
	{"fit", (PyCFunction) GMM_fit, METH_VARARGS | METH_KEYWORDS, "Fit the GMM on the data."},
	{"score", (PyCFunction) GMM_score, METH_VARARGS | METH_KEYWORDS, "Scores the data using the GMM."},
	{NULL}
};

static PyTypeObject GMMObjectType = {
	PyObject_HEAD_INIT(NULL)
	0,						 	/*ob_size*/
	"gmm.GMM",					/*tp_name*/
	sizeof(GMMObject),			/*tp_basicsize*/
	0,							/*tp_itemsize*/
	(destructor) GMM_dealloc,	/*tp_dealloc*/
	0,							/*tp_print*/
	0,							/*tp_getattr*/
	0,							/*tp_setattr*/
	0,							/*tp_compare*/
	0,							/*tp_repr*/
	0,							/*tp_as_number*/
	0,							/*tp_as_sequence*/
	0,							/*tp_as_mapping*/
	0,							/*tp_hash */
	0,							/*tp_call*/
	0,							/*tp_str*/
	0,							/*tp_getattro*/
	0,							/*tp_setattro*/
	0,							/*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,			/*tp_flags*/
	"GMM Object",				/* tp_doc */
	0,							/*tp_traverse*/
	0,							/*tp_clear*/
	0,							/*tp_richcompare*/
	0,							/*tp_weaklistoffset*/
	0,							/*tp_iter*/
	0,							/*tp_iternext*/
	GMM_methods,				/*tp_methods*/
	GMM_members,				/*tp_members*/
	0,							/*tp_getset*/
	0,							/*tp_base*/
	0,							/*tp_dict*/
	0,							/*tp_descr_get*/
	0,							/*tp_descr_set*/
	0,							/*tp_dictoffset*/
	(initproc) GMM_init,		/*tp_init*/
};

static PyMethodDef gmmMethods[] =
{
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initgmm(void)
{
	PyObject *m = Py_InitModule("gmm", gmmMethods);
	if (m == NULL)
	{
		printf("Module object is null\n");
		return;
	}

	import_array();		// For numpy functionality
	
	GMMObjectType.tp_new = PyType_GenericNew;
	if (PyType_Ready(&GMMObjectType) < 0)
		return;
	Py_INCREF(&GMMObjectType);
    PyModule_AddObject(m, "GMM", (PyObject *) &GMMObjectType);
}
