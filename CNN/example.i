/* File : example.i */
%module CNN

%{
#include "example.h"
#include "Layer.h"
#include "SequentialModel.h"
#include "Conv2DLayer.h"
#include "DenseLayer.h"
#include "Pool2DLayer.h"
%}

/* Let's just grab the original header file here */
%include "example.h"
%include "Layer.h"
%include "SequentialModel.h"
%include "Conv2DLayer.h"
%include "DenseLayer.h"
%include "Pool2DLayer.h"