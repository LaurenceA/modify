import inspect
from numbers import Number

import torch
import torch.nn as nn

class _ModifyModule(nn.Module):
    pass

class ModifyModuleGroup(_ModifyModule):
    def __init__(self, mods):
        super().__init__()
        #mods must be a list, dict or tuple
        if not isinstance(mods, (dict, list, tuple)):
            raise Exception(f"Argument to Sequential or Parallel must be dict, list or tuple, but it is {type(mods)}")

        if isinstance(mods, dict):
            self.mods = mods
        else:
            #mods is a list or tuple.
            self.mods = {}
            for i, mod in enumerate(mods):
                if isinstance(mod, tuple):
                    if (not len(mod)==2) and isinstance(mod[1], str):
                        raise Exception("If you give a list a tuple, it should be a tuple with two elements, with a string name at the start")
                    module_name, module = mod
                else:
                    module_name = str(i)
                    module = mod

                self.mods[module_name] = module

        for mod in self.mods.values():
            if not isinstance(mod, ModifyModule):
                raise Exception("Module not recognised (not a subtype of ModifyModule). If you're trying to use a batchnorm module, you need to use the modify wrapper (e.g. torch.nn.BatchNorm1d -> modify.BatchNorm1d), which gives a parameter-free batchnorm. Use an explicit modify.Affine layer afterwards if you want that.")

        self.module_list = nn.ModuleList(self.mods.values())

    def __getitem__(self, key):
        if not isinstance(key, (str, int)):
            raise Exception("Key must be integer or string")

        if isinstance(key, int):
            key = str(key)

        return self.mods[key]

    def state_dict(self, *args, **kwargs):
        result = {}
        for child_name, child_mod in self.mods.items():
            child_state_dict = child_mod.state_dict()
            child_state_dict = {f'{child_name}.{k}': v for (k, v) in child_state_dict.items()}
            result = {**result, **child_state_dict}
        return result

    def load_state_dict(self, state_dict, *args, **kwargs):
        nested_state_dict = {k: {} for k in self.mods.keys()}
        for key, val in state_dict.items():
            head, rest = key.split('.', 1)
            nested_state_dict[head][rest] = val

        for child_name, child_mod in self.mods.items():
            child_mod.load_state_dict(nested_state_dict[child_name])

    def __repr__(self):
        return '\n'.join(repr_lines(self))

def repr_lines(mod, name="", prefix=""):
    lines = []
    if name:
        name = '"' + name + '": '
    if isinstance(mod, ModifyModuleGroup):
        lines.append(prefix + name + mod.__class__.__name__ + '({')
        for child_name, child_module in mod.mods.items():
            lines = lines + repr_lines(child_module, name=child_name, prefix=prefix + '  ')
        lines.append(prefix + '}),')
    else:
        lines = [prefix + name + repr(mod) + ',']
    return lines
        
            
        

class Sequential(ModifyModuleGroup):
    """
    Takes a list, tuple or dict of modules as an input.
    Applies each module to the input in sequence.
    """
    def __call__(self, x):
        validate_tuple_or_tensor(x)
        for mod in self.mods.values():
            x = mod(x)
        return x

class Parallel(ModifyModuleGroup):
    """
    Takes a list, tuple or dict of modules as an input.
    Takes a tuple of input, applies one module to the corresponding input.
    """
    def __call__(self, xs):
        validate_tuple(xs)

        if not len(xs) == len(self.mods):
            raise Exception(f"Parallel expected {len(self.mods)} inputs, but got {len(self.mods)} inputs")
        return tuple(mod(x) for (mod, x) in zip(self.module_list, xs))
    
def validate_tuple(xs):
    if not isinstance(xs, tuple):
        raise Exception(f"Input must be tuple, but actually {type(xs)}")
    else:
        for x in xs:
            validate_tuple_or_tensor(x)

def validate_tensor(x):
    if not isinstance(x, (Number, torch.Tensor)):
        raise Exception(f"Input must be single tensor, but actually {type(xs)}")

def validate_tuple_or_tensor(xs):
    if not isinstance(xs, (tuple, Number, torch.Tensor)):
        raise Exception("Input was not a tuple, Number, or torch.Tensor, instead it was {type(xs)}")
    elif isinstance(xs, tuple):
        for x in xs:
            validate_tuple_or_tensor(x)
    

class Property(_ModifyModule):
    #Concrete subclasses override property_name
    def __call__(self, x):
        validate_tensor(x)
        return getattr(x, self.property_name)

    def __repr__(self):
        return f'{self.__class__.__name__}'



class MethodFunction(_ModifyModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
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


class Method(MethodFunction):
    #Concrete subclasses override method_name
    def __call__(self, x):
        validate_tensor(x)
        return getattr(x, self.method_name)(*self.args, **self.kwargs)

class Function(MethodFunction): pass 
    #Concrete subclasses override function_name

class MultipleArgFunction(Function):
    def __call__(self, xs):
        validate_tuple(xs)
        return self.function(*xs, *self.args, **self.kwargs)

class TupleArgFunction(Function):
    def __call__(self, xs):
        validate_tuple(xs)
        return self.function(xs, *self.args, **self.kwargs)


#Classes that describe the type of operation
class _Restructure: pass #Something like view, that just shuffles around elements of the input.
class _Elementwise: pass #Operates on elements of the input independently.
class _Vector: pass #Operates on vectors of the input independently.  Must have a dim argument.
class _Matrix: pass #Operates on matrices in the last two dimensions independently.
class _Reduction: pass #Reduces a dimension.  Must have a dim and a keepdim argument.

#Classes that describe whether the module has parameters
class _Param: pass
class _ParamFree: pass

#Classes that describe Linear vs NonLinear
class _NonLinear: pass
class _Linear: pass

#Classes that describe the input args
class _Unary: pass #Takes one argument.
class _Binary: pass #Takes two arguments.
class _Ternary: pass #Takes three arguments.
class _Tuple: pass #Takes a tuple of arguments.

restructure_properties = [
'H',
'T',
'mT',
'mH',
]

for property_name in restructure_properties:
    new_class = type(property_name, (Property,_Unary,_Matrix,_Restructure,_ParamFree,_Linear), {
        'property_name': property_name
    })
    globals()[property_name] = new_class

unary_restructure_methods = {
    'adjoint': 'Adjoint',
    'align_as': 'AlignAs',
    'align_to': 'AlignTo',
    'chunk': 'Chunk',
    'conj': 'Conj',
    'diag': 'Diag',
    'diag_embed': 'DiagEmbed',
    'diag_flat': 'DiagFlat',
    'diagonal': 'Diagonal',
    'diagonal_scatter': 'DiagonalScatter',
    'diff': 'Diff',
    'flip': 'Flip',
    'fliplr': 'FlipLR',
    'flipud': 'FlipUD',
    'flatten': 'Flatten',
    'imag': 'Imag',
    'real': 'Real',
    'moveaxis': 'MoveAxis',
    'movedim': 'MoveDim',
    'permute': 'Permute',
    'repeat': 'Repeat',
    'reshape': 'Reshape',
    'resize': 'Resize',
    'rot90': 'Rot90',
    'roll': 'Roll',
    'squeeze': 'Squeeze',
    'vsplit': 'VSplit',
    'hsplit': 'HSplit',
    'split': 'Split',
    't': 'T',
    'tensor_split': 'TensorSplit',
    'transpose': 'Transpose',
    'tril': 'TriL',
    'triu': 'TriU',
    'view': 'View',
    'view_as': 'ViewAs',
}

for method_name, class_name in unary_restructure_methods.items():
    new_class = type(class_name, (Method,_Unary,_Restructure,_ParamFree,_Linear), {
        'method_name': method_name
    })
    globals()[class_name] = new_class

tuple_restructure_functions = {
    'cat': torch.cat,
    'concatenate': torch.concatenate,
    'stack': torch.stack,
    'vstack': torch.vstack,
    'hstack': torch.hstack,
}

for class_name, function in tuple_restructure_functions.items():
    new_class = type(class_name, (TupleArgFunction,_Tuple,_Restructure,_ParamFree,_Linear), {
        'function': function
    })
    globals()[class_name] = new_class

nonlin_reduction_methods = {
    'prod': 'Prod',
    'amax': 'Max',
    'amin': 'Min',
    'argmax': 'ArgMax',
    'argmin': 'ArgMin',
    'std' : 'Std',
    'logsumexp': 'LogSumExp',
    'var': 'Var',
}

for method_name, class_name in nonlin_reduction_methods.items():
    new_class = type(class_name, (Method,_Unary,_Reduction,_ParamFree,_NonLinear), {
        'method_name': method_name
    })
    globals()[class_name] = new_class

lin_reduction_methods = {
    'mean': 'Mean',
    'sum': 'Sum',
}

for method_name, class_name in lin_reduction_methods.items():
    new_class = type(class_name, (Method,_Unary,_Reduction,_ParamFree,_Linear), {
        'method_name': method_name
    })
    globals()[class_name] = new_class

nonlin_unary_vector_methods = {
    'cumprod': 'CumProd',
    'cummax': 'CumMax',
    'cummin': 'CumMin',
    'logcumsumexp': 'LogCumSumExp',
    'logsoftmax': 'LogSoftmax',
    'softmax': 'Softmax',
}

for method_name, class_name in nonlin_unary_vector_methods.items():
    new_class = type(class_name, (Method,_Unary,_Vector,_ParamFree,_NonLinear), {
        'method_name': method_name
    })
    globals()[class_name] = new_class


lin_unary_vector_methods = {
    'cumsum': 'CumSum',
}

for method_name, class_name in lin_unary_vector_methods.items():
    new_class = type(class_name, (Method,_Unary,_Vector,_ParamFree,_Linear), {
        'method_name': method_name
    })
    globals()[class_name] = new_class

unary_matrix_methods = {
    'inverse': 'Inverse',
    'logdet': 'LogDet',
    'matrix_exp': 'MatrixExp',
    'matrix_power': 'MatrixPower',
    'qr': 'QR',
    'slogdet': 'SLogDet',
    'logdet': 'LogDet',
    'det': 'Det',
}

for method_name, class_name in unary_matrix_methods.items():
    new_class = type(class_name, (Method,_Unary,_Matrix,_ParamFree,_NonLinear), {
        'method_name': method_name
    })
    globals()[class_name] = new_class

nonlin_unary_elementwise_methods = {
    'abs': 'Abs',
    'absolute': 'Absolute',
    'acos': 'ACos',
    'acosh': 'ACosh',
    'angle': 'Angle',
    'acos': 'ACos',
    'acosh': 'ACosh',
    'asin': 'ASin',
    'asinh': 'ASinh',
    'atan': 'ATan',
    'atanh': 'ATanh',
    'cos': 'Cos',
    'cosh': 'Cosh',
    'eig': 'Eig',
    'erf': 'Erf',
    'erfc': 'Erfc',
    'erfinv': 'ErfInv',
    'exp': 'Exp',
    'exp2': 'Exp2',
    'expm1': 'Expm1',
    'sin': 'Sin',
    'sinh': 'Sinh',
    'sinc': 'Sinc',
    'tan': 'Tan',
    'tanh': 'Tanh',
    'digamma': 'DiGamma',
    'heaviside': 'Heaviside',
    'igamma': 'IGamma',
    'igammac': 'IGammaC',
    'lgamma': 'LGamma',
    'log': 'Log',
    'log10': 'Log10',
    'log1p': 'Log1p',
    'log2': 'Log2',
    'reciprocal': 'Reciprocal',
    'rsqrt': 'RSqrt',
}

for method_name, class_name in nonlin_unary_elementwise_methods.items():
    new_class = type(class_name, (Method,_Unary, _Elementwise,_ParamFree,_NonLinear), {
        'method_name': method_name
    })
    globals()[class_name] = new_class

lin_unary_elementwise_methods = {
    'negative': 'Negative',
}

for method_name, class_name in lin_unary_elementwise_methods.items():
    new_class = type(class_name, (Method,_Unary, _Elementwise,_ParamFree,_Linear), {
        'method_name': method_name
    })
    globals()[class_name] = new_class


nonlin_binary_elementwise_functions = {
    'Div': torch.div,
    'Mul': torch.mul,
    'Maximum': torch.maximum,
    'Minimum': torch.minimum,
}
for class_name, function in nonlin_binary_elementwise_functions.items():
    new_class = type(class_name, (MultipleArgFunction,_Binary,_Elementwise,_ParamFree,_NonLinear), {
        'function': function
    })
    globals()[class_name] = new_class

lin_binary_elementwise_functions = {
    'Add': torch.add, #Linear
    'Sub': torch.sub, #Linear
}
for class_name, function in lin_binary_elementwise_functions.items():
    new_class = type(class_name, (MultipleArgFunction,_Binary,_Elementwise,_ParamFree,_Linear), {
        'function': function
    })
    globals()[class_name] = new_class

norms = [
'BatchNorm1d',
'BatchNorm2d',
'BatchNorm3d',
'GroupNorm',
'SyncBatchNorm',
'InstanceNorm1d',
'InstanceNorm2d',
'InstanceNorm3d',
'LayerNorm',
'RMSNorm',
]

class Norm(_ModifyModule,_Unary,_ParamFree,_NonLinear):
    def __init__(self, *args, **kwargs):
        super().__init__()
        sig = inspect.signature(self._class.__init__)

        parameters = [*sig.parameters.keys()]
        assert ('affine' in parameters) or ('elementwise_affine' in parameters)
        if 'affine' in parameters:
            affine_argument_name = 'affine'
        else:
            affine_argument_name = 'elementwise_affine'

        bound_args = sig.bind(None, *args, **kwargs).arguments
        if ('affine' in bound_args) or ('elementwise_affine' in bound_args):
            raise Exception(f"You cannot provide affine or elementwise_affine args; these are set to False by Modify so that normalization layers are parameter free. If you want an Affine transformation, then you can include one explicitly using modify.Affine")
        kwargs = {**kwargs, affine_argument_name: False}
        self._mod = self._class(*args, **kwargs)

    def forward(self, x):
        return self._mod(x)

    def state_dict(self, *args, **kwargs):
        return self._mod.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._mod.load_state_dict(*args, **kwargs)

    def __repr__(self):
        return self._mod.__repr__()

for norm_name in norms:
    new_class = type(norm_name, (Norm,_Unary,_ParamFree,_NonLinear), {
        '_class' : getattr(torch.nn, norm_name)
    })
    globals()[norm_name] = new_class


#####################
#### New classes ####
#####################

class ElementwiseAffine(_ModifyModule,_Unary,_Elementwise,_Param,_Linear):
    """
    Does elementwise multiplication and biasing (e.g. usually used after norm layers)
    """
    def __init__(self, shape, bias=True):
        super().__init__()
        self.weights = nn.Parameter(t.ones(shape))
        self.bias = nn.Parameter(t.zeros(shape)) if bias else None

    def forward(self, x):
        result = self.weights * x
        if bias is not None:
            result = result + bias
        return result

class Copy(_ModifyModule,_Unary,_Elementwise,_ParamFree,_Linear):
    """
    Takes a single input (e.g. a Tensor), and copies it to form a tuple of copies of that Tensor.
    Useful as input to nn.Parallel, e.g. for a residual block
    """
    def __init__(self, number_of_copies):
        super().__init__()
        self.number_of_copies = number_of_copies

    def forward(self, x):
        return self.number_of_copies * (x,)

class Debug(_ModifyModule,_ParamFree,_Linear): pass

class BreakPoint(Debug):
    def forward(self, x):
        validate_tuple_or_tensor(x)
        breakpoint()
        return x

class DebugPrint(Debug):
    def __init__(self, prefix=""):
        super().__init__()
        if prefix:
            prefix = prefix + ': '
        self.prefix = prefix

class Print(DebugPrint):
    def forward(self, x):
        validate_tuple_or_tensor(x)
        print(f'{self.prefix}{x.__getitem__(index)}')
        return x

def print_shape(xs, indent=""):
    if isinstance(xs, tuple):
        print(f'{indent}(')
        for x in xs:
            print_shape(x, indent=indent+"  ")
        print(f'{indent})')
    else:
        print(f'{indent}shape: {list(xs.shape)}, dtype: {xs.dtype}, device: {xs.device}')

class PrintShape(DebugPrint):
    def forward(self, x):
        print(f'{self.prefix}', end="")
        validate_tuple_or_tensor(x)
        print_shape(x)
        return x
            

TorchLearnedLinear = (
    nn.Linear, 
    nn.Conv1d, 
    nn.Conv2d, 
    nn.Conv3d, 
    nn.ConvTranspose1d, 
    nn.ConvTranspose2d, 
    nn.ConvTranspose3d
)

TorchActivations = (
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


#Classes understood by Modify
ModifyModule = (_ModifyModule, *TorchActivations, *TorchLearnedLinear, nn.Identity)
    
#Classes that describe the type of operation
Restructure = _Restructure
Elementwise = (_Elementwise, *TorchActivations, nn.Identity)
Vector = _Vector
Matrix = _Matrix
Reduction = _Reduction

#Classes that describe whether the module has parameters
Param = (_Param, *TorchLearnedLinear)
ParamFree = (_ParamFree, *TorchActivations, nn.Identity)

#Classes that describe Linear vs NonLinear
NonLinear = (_NonLinear, *TorchActivations)
Linear = (_Linear, *TorchLearnedLinear, nn.Identity)

#Classes that describe the input args
Unary = (_NonLinear, *TorchActivations, *TorchLearnedLinear, nn.Identity)
Binary = _Binary
Ternary = _Ternary
Tuple = _Tuple
