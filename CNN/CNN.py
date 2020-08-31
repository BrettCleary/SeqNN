# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_CNN')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_CNN')
    _CNN = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_CNN', [dirname(__file__)])
        except ImportError:
            import _CNN
            return _CNN
        try:
            _mod = imp.load_module('_CNN', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _CNN = swig_import_helper()
    del swig_import_helper
else:
    import _CNN
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0

class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _CNN.delete_SwigPyIterator
    __del__ = lambda self: None

    def value(self):
        return _CNN.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _CNN.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _CNN.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _CNN.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _CNN.SwigPyIterator_equal(self, x)

    def copy(self):
        return _CNN.SwigPyIterator_copy(self)

    def next(self):
        return _CNN.SwigPyIterator_next(self)

    def __next__(self):
        return _CNN.SwigPyIterator___next__(self)

    def previous(self):
        return _CNN.SwigPyIterator_previous(self)

    def advance(self, n):
        return _CNN.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _CNN.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _CNN.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _CNN.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _CNN.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _CNN.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _CNN.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self
SwigPyIterator_swigregister = _CNN.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class doubleVector1D(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, doubleVector1D, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, doubleVector1D, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _CNN.doubleVector1D_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _CNN.doubleVector1D___nonzero__(self)

    def __bool__(self):
        return _CNN.doubleVector1D___bool__(self)

    def __len__(self):
        return _CNN.doubleVector1D___len__(self)

    def __getslice__(self, i, j):
        return _CNN.doubleVector1D___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _CNN.doubleVector1D___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _CNN.doubleVector1D___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _CNN.doubleVector1D___delitem__(self, *args)

    def __getitem__(self, *args):
        return _CNN.doubleVector1D___getitem__(self, *args)

    def __setitem__(self, *args):
        return _CNN.doubleVector1D___setitem__(self, *args)

    def pop(self):
        return _CNN.doubleVector1D_pop(self)

    def append(self, x):
        return _CNN.doubleVector1D_append(self, x)

    def empty(self):
        return _CNN.doubleVector1D_empty(self)

    def size(self):
        return _CNN.doubleVector1D_size(self)

    def swap(self, v):
        return _CNN.doubleVector1D_swap(self, v)

    def begin(self):
        return _CNN.doubleVector1D_begin(self)

    def end(self):
        return _CNN.doubleVector1D_end(self)

    def rbegin(self):
        return _CNN.doubleVector1D_rbegin(self)

    def rend(self):
        return _CNN.doubleVector1D_rend(self)

    def clear(self):
        return _CNN.doubleVector1D_clear(self)

    def get_allocator(self):
        return _CNN.doubleVector1D_get_allocator(self)

    def pop_back(self):
        return _CNN.doubleVector1D_pop_back(self)

    def erase(self, *args):
        return _CNN.doubleVector1D_erase(self, *args)

    def __init__(self, *args):
        this = _CNN.new_doubleVector1D(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _CNN.doubleVector1D_push_back(self, x)

    def front(self):
        return _CNN.doubleVector1D_front(self)

    def back(self):
        return _CNN.doubleVector1D_back(self)

    def assign(self, n, x):
        return _CNN.doubleVector1D_assign(self, n, x)

    def resize(self, *args):
        return _CNN.doubleVector1D_resize(self, *args)

    def insert(self, *args):
        return _CNN.doubleVector1D_insert(self, *args)

    def reserve(self, n):
        return _CNN.doubleVector1D_reserve(self, n)

    def capacity(self):
        return _CNN.doubleVector1D_capacity(self)
    __swig_destroy__ = _CNN.delete_doubleVector1D
    __del__ = lambda self: None
doubleVector1D_swigregister = _CNN.doubleVector1D_swigregister
doubleVector1D_swigregister(doubleVector1D)

class doubleVector2D(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, doubleVector2D, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, doubleVector2D, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _CNN.doubleVector2D_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _CNN.doubleVector2D___nonzero__(self)

    def __bool__(self):
        return _CNN.doubleVector2D___bool__(self)

    def __len__(self):
        return _CNN.doubleVector2D___len__(self)

    def __getslice__(self, i, j):
        return _CNN.doubleVector2D___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _CNN.doubleVector2D___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _CNN.doubleVector2D___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _CNN.doubleVector2D___delitem__(self, *args)

    def __getitem__(self, *args):
        return _CNN.doubleVector2D___getitem__(self, *args)

    def __setitem__(self, *args):
        return _CNN.doubleVector2D___setitem__(self, *args)

    def pop(self):
        return _CNN.doubleVector2D_pop(self)

    def append(self, x):
        return _CNN.doubleVector2D_append(self, x)

    def empty(self):
        return _CNN.doubleVector2D_empty(self)

    def size(self):
        return _CNN.doubleVector2D_size(self)

    def swap(self, v):
        return _CNN.doubleVector2D_swap(self, v)

    def begin(self):
        return _CNN.doubleVector2D_begin(self)

    def end(self):
        return _CNN.doubleVector2D_end(self)

    def rbegin(self):
        return _CNN.doubleVector2D_rbegin(self)

    def rend(self):
        return _CNN.doubleVector2D_rend(self)

    def clear(self):
        return _CNN.doubleVector2D_clear(self)

    def get_allocator(self):
        return _CNN.doubleVector2D_get_allocator(self)

    def pop_back(self):
        return _CNN.doubleVector2D_pop_back(self)

    def erase(self, *args):
        return _CNN.doubleVector2D_erase(self, *args)

    def __init__(self, *args):
        this = _CNN.new_doubleVector2D(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _CNN.doubleVector2D_push_back(self, x)

    def front(self):
        return _CNN.doubleVector2D_front(self)

    def back(self):
        return _CNN.doubleVector2D_back(self)

    def assign(self, n, x):
        return _CNN.doubleVector2D_assign(self, n, x)

    def resize(self, *args):
        return _CNN.doubleVector2D_resize(self, *args)

    def insert(self, *args):
        return _CNN.doubleVector2D_insert(self, *args)

    def reserve(self, n):
        return _CNN.doubleVector2D_reserve(self, n)

    def capacity(self):
        return _CNN.doubleVector2D_capacity(self)
    __swig_destroy__ = _CNN.delete_doubleVector2D
    __del__ = lambda self: None
doubleVector2D_swigregister = _CNN.doubleVector2D_swigregister
doubleVector2D_swigregister(doubleVector2D)

class doubleVector3D(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, doubleVector3D, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, doubleVector3D, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _CNN.doubleVector3D_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _CNN.doubleVector3D___nonzero__(self)

    def __bool__(self):
        return _CNN.doubleVector3D___bool__(self)

    def __len__(self):
        return _CNN.doubleVector3D___len__(self)

    def __getslice__(self, i, j):
        return _CNN.doubleVector3D___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _CNN.doubleVector3D___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _CNN.doubleVector3D___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _CNN.doubleVector3D___delitem__(self, *args)

    def __getitem__(self, *args):
        return _CNN.doubleVector3D___getitem__(self, *args)

    def __setitem__(self, *args):
        return _CNN.doubleVector3D___setitem__(self, *args)

    def pop(self):
        return _CNN.doubleVector3D_pop(self)

    def append(self, x):
        return _CNN.doubleVector3D_append(self, x)

    def empty(self):
        return _CNN.doubleVector3D_empty(self)

    def size(self):
        return _CNN.doubleVector3D_size(self)

    def swap(self, v):
        return _CNN.doubleVector3D_swap(self, v)

    def begin(self):
        return _CNN.doubleVector3D_begin(self)

    def end(self):
        return _CNN.doubleVector3D_end(self)

    def rbegin(self):
        return _CNN.doubleVector3D_rbegin(self)

    def rend(self):
        return _CNN.doubleVector3D_rend(self)

    def clear(self):
        return _CNN.doubleVector3D_clear(self)

    def get_allocator(self):
        return _CNN.doubleVector3D_get_allocator(self)

    def pop_back(self):
        return _CNN.doubleVector3D_pop_back(self)

    def erase(self, *args):
        return _CNN.doubleVector3D_erase(self, *args)

    def __init__(self, *args):
        this = _CNN.new_doubleVector3D(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _CNN.doubleVector3D_push_back(self, x)

    def front(self):
        return _CNN.doubleVector3D_front(self)

    def back(self):
        return _CNN.doubleVector3D_back(self)

    def assign(self, n, x):
        return _CNN.doubleVector3D_assign(self, n, x)

    def resize(self, *args):
        return _CNN.doubleVector3D_resize(self, *args)

    def insert(self, *args):
        return _CNN.doubleVector3D_insert(self, *args)

    def reserve(self, n):
        return _CNN.doubleVector3D_reserve(self, n)

    def capacity(self):
        return _CNN.doubleVector3D_capacity(self)
    __swig_destroy__ = _CNN.delete_doubleVector3D
    __del__ = lambda self: None
doubleVector3D_swigregister = _CNN.doubleVector3D_swigregister
doubleVector3D_swigregister(doubleVector3D)

class doubleVector4D(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, doubleVector4D, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, doubleVector4D, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _CNN.doubleVector4D_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _CNN.doubleVector4D___nonzero__(self)

    def __bool__(self):
        return _CNN.doubleVector4D___bool__(self)

    def __len__(self):
        return _CNN.doubleVector4D___len__(self)

    def __getslice__(self, i, j):
        return _CNN.doubleVector4D___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _CNN.doubleVector4D___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _CNN.doubleVector4D___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _CNN.doubleVector4D___delitem__(self, *args)

    def __getitem__(self, *args):
        return _CNN.doubleVector4D___getitem__(self, *args)

    def __setitem__(self, *args):
        return _CNN.doubleVector4D___setitem__(self, *args)

    def pop(self):
        return _CNN.doubleVector4D_pop(self)

    def append(self, x):
        return _CNN.doubleVector4D_append(self, x)

    def empty(self):
        return _CNN.doubleVector4D_empty(self)

    def size(self):
        return _CNN.doubleVector4D_size(self)

    def swap(self, v):
        return _CNN.doubleVector4D_swap(self, v)

    def begin(self):
        return _CNN.doubleVector4D_begin(self)

    def end(self):
        return _CNN.doubleVector4D_end(self)

    def rbegin(self):
        return _CNN.doubleVector4D_rbegin(self)

    def rend(self):
        return _CNN.doubleVector4D_rend(self)

    def clear(self):
        return _CNN.doubleVector4D_clear(self)

    def get_allocator(self):
        return _CNN.doubleVector4D_get_allocator(self)

    def pop_back(self):
        return _CNN.doubleVector4D_pop_back(self)

    def erase(self, *args):
        return _CNN.doubleVector4D_erase(self, *args)

    def __init__(self, *args):
        this = _CNN.new_doubleVector4D(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _CNN.doubleVector4D_push_back(self, x)

    def front(self):
        return _CNN.doubleVector4D_front(self)

    def back(self):
        return _CNN.doubleVector4D_back(self)

    def assign(self, n, x):
        return _CNN.doubleVector4D_assign(self, n, x)

    def resize(self, *args):
        return _CNN.doubleVector4D_resize(self, *args)

    def insert(self, *args):
        return _CNN.doubleVector4D_insert(self, *args)

    def reserve(self, n):
        return _CNN.doubleVector4D_reserve(self, n)

    def capacity(self):
        return _CNN.doubleVector4D_capacity(self)
    __swig_destroy__ = _CNN.delete_doubleVector4D
    __del__ = lambda self: None
doubleVector4D_swigregister = _CNN.doubleVector4D_swigregister
doubleVector4D_swigregister(doubleVector4D)

class outputVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, outputVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, outputVector, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _CNN.outputVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _CNN.outputVector___nonzero__(self)

    def __bool__(self):
        return _CNN.outputVector___bool__(self)

    def __len__(self):
        return _CNN.outputVector___len__(self)

    def __getslice__(self, i, j):
        return _CNN.outputVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _CNN.outputVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _CNN.outputVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _CNN.outputVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _CNN.outputVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _CNN.outputVector___setitem__(self, *args)

    def pop(self):
        return _CNN.outputVector_pop(self)

    def append(self, x):
        return _CNN.outputVector_append(self, x)

    def empty(self):
        return _CNN.outputVector_empty(self)

    def size(self):
        return _CNN.outputVector_size(self)

    def swap(self, v):
        return _CNN.outputVector_swap(self, v)

    def begin(self):
        return _CNN.outputVector_begin(self)

    def end(self):
        return _CNN.outputVector_end(self)

    def rbegin(self):
        return _CNN.outputVector_rbegin(self)

    def rend(self):
        return _CNN.outputVector_rend(self)

    def clear(self):
        return _CNN.outputVector_clear(self)

    def get_allocator(self):
        return _CNN.outputVector_get_allocator(self)

    def pop_back(self):
        return _CNN.outputVector_pop_back(self)

    def erase(self, *args):
        return _CNN.outputVector_erase(self, *args)

    def __init__(self, *args):
        this = _CNN.new_outputVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _CNN.outputVector_push_back(self, x)

    def front(self):
        return _CNN.outputVector_front(self)

    def back(self):
        return _CNN.outputVector_back(self)

    def assign(self, n, x):
        return _CNN.outputVector_assign(self, n, x)

    def resize(self, *args):
        return _CNN.outputVector_resize(self, *args)

    def insert(self, *args):
        return _CNN.outputVector_insert(self, *args)

    def reserve(self, n):
        return _CNN.outputVector_reserve(self, n)

    def capacity(self):
        return _CNN.outputVector_capacity(self)
    __swig_destroy__ = _CNN.delete_outputVector
    __del__ = lambda self: None
outputVector_swigregister = _CNN.outputVector_swigregister
outputVector_swigregister(outputVector)

class Layer(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Layer, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Layer, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr

    def ResetWeights(self):
        return _CNN.Layer_ResetWeights(self)

    def GradientCorrect(self, model, startIndex, endIndex):
        return _CNN.Layer_GradientCorrect(self, model, startIndex, endIndex)

    def UpdateWeights(self):
        return _CNN.Layer_UpdateWeights(self)

    def GetOutput(self):
        return _CNN.Layer_GetOutput(self)

    def FwdProp(self, input):
        return _CNN.Layer_FwdProp(self, input)

    def BackProp(self, backPropErrorSum):
        return _CNN.Layer_BackProp(self, backPropErrorSum)

    def GetName(self):
        return _CNN.Layer_GetName(self)
    __swig_destroy__ = _CNN.delete_Layer
    __del__ = lambda self: None
Layer_swigregister = _CNN.Layer_swigregister
Layer_swigregister(Layer)

class SequentialModel(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SequentialModel, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SequentialModel, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = _CNN.new_SequentialModel()
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def GetInputDataPointsVector(self):
        return _CNN.SequentialModel_GetInputDataPointsVector(self)

    def GetTargetVectors(self):
        return _CNN.SequentialModel_GetTargetVectors(self)

    def CheckGradientNumerically(self):
        return _CNN.SequentialModel_CheckGradientNumerically(self)

    def CalcErrorNumerically(self, dataPointIndex):
        return _CNN.SequentialModel_CalcErrorNumerically(self, dataPointIndex)

    def AddInputDataPoint(self, len1_):
        return _CNN.SequentialModel_AddInputDataPoint(self, len1_)

    def AddInputDataPoints(self, len1_):
        return _CNN.SequentialModel_AddInputDataPoints(self, len1_)

    def AddTargetVector(self, len1_):
        return _CNN.SequentialModel_AddTargetVector(self, len1_)

    def AddTargetVectors(self, len1_):
        return _CNN.SequentialModel_AddTargetVectors(self, len1_)

    def SetBatchSize(self, sz):
        return _CNN.SequentialModel_SetBatchSize(self, sz)

    def SetNumEpochs(self, num):
        return _CNN.SequentialModel_SetNumEpochs(self, num)

    def AddLayer(self, layer):
        return _CNN.SequentialModel_AddLayer(self, layer)

    def ClearLayers(self):
        return _CNN.SequentialModel_ClearLayers(self)

    def Train(self):
        return _CNN.SequentialModel_Train(self)

    def Predict(self, len1_):
        return _CNN.SequentialModel_Predict(self, len1_)
    __swig_destroy__ = _CNN.delete_SequentialModel
    __del__ = lambda self: None
SequentialModel_swigregister = _CNN.SequentialModel_swigregister
SequentialModel_swigregister(SequentialModel)

class Conv2DLayer(Layer):
    __swig_setmethods__ = {}
    for _s in [Layer]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Conv2DLayer, name, value)
    __swig_getmethods__ = {}
    for _s in [Layer]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Conv2DLayer, name)
    __repr__ = _swig_repr

    def __init__(self, winRows, winCols, strideRowInput, strideColInput, paddingInput, step, momentum):
        this = _CNN.new_Conv2DLayer(winRows, winCols, strideRowInput, strideColInput, paddingInput, step, momentum)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def FwdProp(self, input):
        return _CNN.Conv2DLayer_FwdProp(self, input)

    def BackProp(self, backPropErrorSum):
        return _CNN.Conv2DLayer_BackProp(self, backPropErrorSum)
    __swig_destroy__ = _CNN.delete_Conv2DLayer
    __del__ = lambda self: None
Conv2DLayer_swigregister = _CNN.Conv2DLayer_swigregister
Conv2DLayer_swigregister(Conv2DLayer)

logSig = _CNN.logSig
softmax = _CNN.softmax
class DenseLayer(Layer):
    __swig_setmethods__ = {}
    for _s in [Layer]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, DenseLayer, name, value)
    __swig_getmethods__ = {}
    for _s in [Layer]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, DenseLayer, name)
    __repr__ = _swig_repr

    def __init__(self, step, outRows, outCols, momentum, activationFxn):
        this = _CNN.new_DenseLayer(step, outRows, outCols, momentum, activationFxn)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def FwdProp(self, input):
        return _CNN.DenseLayer_FwdProp(self, input)

    def BackProp(self, backPropErrorSum):
        return _CNN.DenseLayer_BackProp(self, backPropErrorSum)
    __swig_destroy__ = _CNN.delete_DenseLayer
    __del__ = lambda self: None
DenseLayer_swigregister = _CNN.DenseLayer_swigregister
DenseLayer_swigregister(DenseLayer)

class Pool2DLayer(Layer):
    __swig_setmethods__ = {}
    for _s in [Layer]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Pool2DLayer, name, value)
    __swig_getmethods__ = {}
    for _s in [Layer]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Pool2DLayer, name)
    __repr__ = _swig_repr

    def __init__(self, isMaxPool, poolColsInput, poolRowsInput):
        this = _CNN.new_Pool2DLayer(isMaxPool, poolColsInput, poolRowsInput)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def FwdProp(self, input):
        return _CNN.Pool2DLayer_FwdProp(self, input)

    def BackProp(self, backPropErrorSum):
        return _CNN.Pool2DLayer_BackProp(self, backPropErrorSum)
    __swig_destroy__ = _CNN.delete_Pool2DLayer
    __del__ = lambda self: None
Pool2DLayer_swigregister = _CNN.Pool2DLayer_swigregister
Pool2DLayer_swigregister(Pool2DLayer)

# This file is compatible with both classic and new-style classes.


