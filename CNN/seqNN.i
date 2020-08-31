/* File : seqNN.i */
%module CNN

%include "std_vector.i"

%{
#define SWIG_FILE_WITH_INIT
#include "Layer.h"
#include "DenseLayer.h"
#include "Pool2DLayer.h"
#include "SequentialModel.h"
%}

namespace std{
#include <vector>

%template(doubleVector1D) vector<double>;
%template(doubleVector2D) vector<vector<double>>;
%template(doubleVector3D) vector<vector<vector<double>>>;
%template(doubleVector4D) vector<vector<vector<vector<double>>>>;
%template() vector<vector<vector<vector<char>>>>;
%template(outputVector) vector<int>;
}

%include "numpy.i"

%init %{
import_array();
%}

//NUMPY Interface Function Signatures
%apply (int DIM1, int DIM2, double* IN_ARRAY2) {(int len1_, int len2_, double* vec_)}
%apply (int DIM1, double* IN_ARRAY1) {(int len1_, double* vec_)}
%apply ( int DIM1, int DIM2, int DIM3, double* IN_ARRAY3 ) {(int len1_, int len2_, int len3_, double* vec_)}

/* Include the original header files here */
%include "Layer.h"
%include "SequentialModel.h"
%include "Conv2DLayer.h"
%include "DenseLayer.h"
%include "Pool2DLayer.h"
