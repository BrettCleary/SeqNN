/* File : example.i */
%module CNN

%include "std_vector.i"

%{
#define SWIG_FILE_WITH_INIT
#include "example.h"
#include "Layer.h"
#include "DenseLayer.h"
#include "Pool2DLayer.h"
//#include "Conv2DLayer.h"
#include "SequentialModel.h"
%}

namespace std{
#include <vector>

	//%template(layerPtrVector) vector<Layer*>;

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





/*
%rename (_AddInputDataPoint) AddInputDataPoint;
%inline %{
int SequentialModel::AddInputDataPoint(size_t len1_, size_t len2_, double* vec_) {
    std::vector< std::vector<double> > v (len1_);
    for (size_t i = 0; i < len1_; ++i) {
        v[i].insert(v[i].end(), vec_ + i*len2_, vec_ + (i+1)*len2_);
    }
    return _AddInputDataPoint(v);
}
%}
*/





/* Let's just grab the original header file here */
%include "example.h"
%include "Layer.h"
%include "SequentialModel.h"
%include "Conv2DLayer.h"
%include "DenseLayer.h"
%include "Pool2DLayer.h"
