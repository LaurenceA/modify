#TODO
#nn.Identity.
#Residual (convenience)
#BatchNorm
#RMSNorm/Layernorm (regard as a linear operation, with fixed parameters).
#Self-attention.
#ModuleGroup behaves like a dict (in, haskey etc).

#Transform modules to capture inputs + backpropagating gradients.
#Linearise forward.
#Linearise reverse.
#Linearising pointwise nonlinearities should be easy, but you need to know how many features...
#Syntax: PointwiseNonLinearity(nn.ReLU / t.sin etc., shape)
#How to linearise 
#How to linearise

import torch
import torch.nn as nn
import inspect
from collections import OrderedDict

collections = (list, dict, tuple)
class ModuleGroup(nn.Module):
    def __init__(self, mods):
        super().__init__()
        #mods must be a list, dict or tuple
        if not isinstance(mods, collections):
            raise Exception(f"Argument to Sequential or Parallel must be list, tuple or dict, but it is {type(mods)}")

        if isinstance(mods, dict):
            self.names = {name: i for (i, name) in enumerate(dict.keys())}
            self.mods = nn.ModuleList(*mods.values())
        else:
            self.names = None
            self.mods = nn.ModuleList(mods)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.mods[key]
        elif isinstance(key, str) and self.names is not None:
            return self.mods[self.names[name]]
        elif isinstance(key, str):
            raise Exception("String key given, but ModuleGroup initialized just with a list/tuple, so modules don't have names")
        else:
            raise Exception("Key must be integer or string")

class Sequential(ModuleGroup):
    def __call__(self, x):
        for mod in self.mods.values():
            x = mod(x)
        return x

class Parallel(ModuleGroup):
    def __call__(self, xs):
        assert set(xs.keys()) == set(self.mods.keys())
        
        result = {}
        for key in self.mods:
            result[key] = self.mods[key](xs[key])
        return result

ParamLinear = (
    nn.Linear, 
    nn.Conv1d, 
    nn.Conv2d, 
    nn.Conv3d, 
    nn.ConvTranspose1d, 
    nn.ConvTranspose2d, 
    nn.ConvTranspose3d
)

TorchNonLinearPointwise = (
    nn.ELU, 
    nn.Hardshrink, 
    nn.Hardsigmoid, 
    nn.Hardtanh,
    nn.Hardswish,
    nn.LeakyReLU,
    nn.LogSigmoid,
    nn.PReLU,
    nn.ReLU,
    nn.ReLU6,
    nn.RReLU,
    nn.SELU,
    nn.CELU,
    nn.GELU,
    nn.Sigmoid,
    nn.SiLU,
    nn.Mish,
    nn.Softplus,
    nn.Softshrink,
    nn.Softsign,
    nn.Tanh,
    nn.Tanhshrink,
    nn.Threshold,
)

TorchNonLinearNonPointwise = (
    nn.MultiheadAttention
)

class ParamFree():
    def parameters(self):
        return []

class Property(ParamFree):
    #Concrete subclasses override property_name
    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            raise Exception("Property should be called with tensor, but actually got {type(x)}")
        return getattr(x, self.property_name)

    def __repr__(self):
        return f'{self.__class__.__name__}'



class MethodFunction(ParamFree):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        # Format positional arguments
        args_str = ", ".join(repr(arg) for arg in self.args)
        
        # Format keyword arguments
        kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in self.kwargs.items())
        
        # Combine args and kwargs
        all_args = ", ".join(filter(bool, [args_str, kwargs_str]))
        
        # Create the function signature
        return f"{self.__class__.__name__}({all_args})"


class Method(ParamFree):
    #Concrete subclasses override method_name
    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            raise Exception("Method should be called with tensor, but actually got {type(x)}")
        return getattr(x, self.method_name)(*self.args, **self.kwargs)

class Function(ParamFree): pass 
    #Concrete subclasses override function_name

class TorchModule(ParamFree): pass

class MultipleArgFunction(Function):
    def __call__(self, xs):
        return self.function(*xs, *self.args, **self.kwargs)

class TupleArgFunction(Function):
    def __call__(self, xs):
        return self.function(xs, *self.args, **self.kwargs)

#Classes of functions:
class ParamFreeLinear(): pass
class NonLinear(): pass
class NonLinearPointwise(NonLinear): pass
class NonLinearNonPointwise(NonLinear): pass

class ParamFreeLinearProperty(ParamFreeLinear, Property): pass
class ParamFreeLinearMethod(ParamFreeLinear, Method): pass
class ParamFreeLinearMultipleArgFunction(ParamFreeLinear, MultipleArgFunction): pass
class ParamFreeLinearTupleArgFunction(ParamFreeLinear, TupleArgFunction): pass

class NonLinearProperty(NonLinear, Property): pass
class NonLinearPointwiseMethod(NonLinearPointwise, Method): pass
class NonLinearNonPointwiseMethod(NonLinearNonPointwise, Method): pass
class NonLinearMultipleArgFunction(NonLinear, MultipleArgFunction): pass

#Classes of function
#ParamLinear
#ParamFreeLinear
#NonLinearPointwise
#NonLinearNonPointwise

linear_properties = [
'H',
'T',
'mT',
'mH',
]

for property_name in linear_properties:
    new_class = type(property_name, (LinearProperty,), {
        'property_name': property_name
    })
    globals()[property_name] = new_class

linear_methods = [
'adjoint',
'align_as',
'align_to',
'chunk',
'cumsum',
'conj',
'diag',
'diag_embed',
'diag_flat',
'diagonal',
'diagonal_scatter',
'diff',
'flip',
'fliplr',
'flipud',
'flatten',
'imag',
'real',
'mean',
'moveaxis',
'movedim',
'permute',
'repeat',
'reshape',
'resize',
'rot90',
'roll',
'squeeze',
'sum',
'vsplit',
'hsplit',
'split',
't',
'tensor_split',
'transpose',
'tril',
'triu',
'view',
'view_as',
]

for method_name in linear_methods:
    new_class = type(method_name.capitalize(), (LinearMethod,), {
        'method_name': method_name
    })
    globals()[method_name.capitalize()] = new_class

nonlinear_nonpointwise_methods = [
'amin',
'amax',
'cummax',
'cummin',
'cumprod',
'inverse',
'logsoftmax',
'logdet',
'logaddexp',
'logaddexp2',
'logcumsumexp',
'logsumexp',
'matrix_exp',
'matrix_power',
'qr',
'slogdet',
'std',
]

for method_name in nonlinear_nonpointwise_methods:
    new_class = type(method_name.capitalize(), (NonLinearNonPointwiseMethod,), {
        'method_name': method_name
    })
    globals()[method_name.capitalize()] = new_class

nonlinear_pointwise_methods = [
'abs',
'absolute',
'acos',
'acosh',
'angle',
'acos',
'acosh',
'asin',
'asinh',
'atan',
'atan2',
'atanh',
'cos',
'cosh',
'eig',
'erf',
'erfc',
'erfinv',
'exp',
'exp2',
'expm1',
'sin',
'sinh',
'sinc',
'tan',
'tanh',
'det',
'digamma',
'heaviside',
'igamma',
'igammac',
'lgamma',
'log',
'log10',
'log1p',
'log2',
'negative',
'reciprocal',
'rsqrt',
]

for method_name in nonlinear_pointwise_methods:
    new_class = type(method_name.capitalize(), (NonLinearPointwiseMethod,), {
        'method_name': method_name
    })
    globals()[method_name.capitalize()] = new_class



linear_multiple_arg_functions = {
'add': torch.add,
'sub': torch.sub,
}

for function_name in linear_multiple_arg_functions:
    new_class = type(function_name.capitalize(), (LinearMultipleArgFunction,), {
        'function': getattr(torch, function_name)
    })
    globals()[function_name.capitalize()] = new_class

linear_tuple_arg_functions = {
'cat': torch.cat,
'concatenate': torch.concatenate,
'stack': torch.stack,
'vstack': torch.vstack,
'hstack': torch.hstack,
}

for function_name in linear_tuple_arg_functions:
    new_class = type(function_name.capitalize(), (LinearTupleArgFunction,), {
        'function': getattr(torch, function_name)
    })
    globals()[function_name.capitalize()] = new_class

nonlinear_functions = {
'div': torch.div,
'dot': torch.dot,
'mul': torch.mul,
'matmul': torch.matmul,
'inner': torch.inner,
'kron': torch.kron,
'maximum': torch.maximum,
'minimum': torch.minimum,
}

for function_name in nonlinear_functions:
    new_class = type(function_name.capitalize(), (NonLinearMultipleArgFunction,), {
        'function': getattr(torch, function_name)
    })
    globals()[function_name.capitalize()] = new_class


