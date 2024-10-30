import inspect
from numbers import Number
from warnings import warn

import torch
import torch.nn as nn

class ModuleGroup(nn.Module):
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
    if isinstance(mod, ModuleGroup):
        lines.append(prefix + name + mod.__class__.__name__ + '({')
        for child_name, child_module in mod.mods.items():
            lines = lines + repr_lines(child_module, name=child_name, prefix=prefix + '  ')
        lines.append(prefix + '}),')
    else:
        lines = [prefix + name + repr(mod) + ',']
    return lines
        
            
        

class Sequential(ModuleGroup):
    """
    Takes a list, tuple or dict of modules as an input.
    Applies each module to the input in sequence.
    """
    def __call__(self, x):
        validate_tuple_or_tensor(x)
        for mod in self.mods.values():
            x = mod(x)
        return x

class Parallel(ModuleGroup):
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



###############################
#### New debugging classes ####
###############################

class Debug(nn.Module): pass

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

class AssertShape(Debug):
    def __init__(self, ndim, shape, dtype, device):
        self.ndim = ndim
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def forward(self, x):
        assert self.ndim == x.ndim
        for (self_shape, x_shape) in zip(self.shape, x.shape):
            if self_shape is not None:
                assert self_shape == x_shape
        assert self.dtype == x.dtype
        assert self.device == x.device

################################
#### New functional classes ####
################################

class Copy(nn.Module):
    """
    Takes a single input (e.g. a Tensor), and copies it to form a tuple of copies of that Tensor.
    Useful as input to nn.Parallel, e.g. for a residual block
    """
    def __init__(self, number_of_copies):
        super().__init__()
        self.number_of_copies = number_of_copies

    def forward(self, x):
        return self.number_of_copies * (x,)

class Add(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, xs):
        assert isinstance(xs, tuple)
        assert len(xs)==2
        return xs[0] + xs[1]

class Mul(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, xs):
        assert isinstance(xs, tuple)
        assert len(xs)==2
        return xs[0] * xs[1]

class ViewMerge(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        assert self.dim0+1 == self.dim1

    def forward(self, x):
        """
        Takes two dimensions that are adjacent, and uses reshape to merge them.
        """
        if dim0 < 0:
            dim0 = x.ndim + self.dim0
        else:
            dim0 = self.dim0

        if dim1 < 0:
            dim1 = x.ndim + self.dim1
        else:
            dim1 = self.dim1

        return x.reshape(*x.shape[:dim0], dim0*dim1, *x.shape[(dim1+1):])

class ViewUnMerge(nn.Module):
    def __init__(self, dim, n0=None, n1=None):
        super().__init__()
        self.dim = dim
        self.n0 = n0
        self.n1 = n1
        assert (n0 is not None) or (n1 is not None)

    def forward(self, x):
        """
        Takes a single dimension, and splits it into two dimensions using view.
        n0 is the size of the first dimension, and n1 is the size of the second dimension.
        if both 
        """
        if self.dim < 0:
            dim = x.ndim + self.dim
        else:
            dim = self.dim

        n0_n1 = x.shape[dim]
        if self.n0 is None:
            assert self.n1 is not None
            self.n0 = n0_n1 // self.n1
        if self.n1 is None:
            assert self.n0 is not None
            self.n1 = n0_n1 // self.n0
            
        if (self.n0 is not None) and (self.n1 is not None):
            assert x.shape[dim] == self.n0*self.n1

        return x.view(*x.shape[:dim], self.n0, self.n1, *x.shape[(dim+1):])

#########################
#### Classes for KLD ####
#########################

class ElementwiseAffine(nn.Module):
    """
    Standard normalization layers combine a parameter-free normalization and
    affine (i.e. bias + scale of the output). Dealing with these together for
    KLD is quite painful.  So you should use elementwise_affine=False, and 
    include an ElementwiseAffine layer afterwards.
    """
    def __init__(self, features, bias=True):
        self.scale = nn.Parameter(t.ones(features))
        self.bias = nn.Parameter(t.ones(features)) if bias else None

    def forward(self, x):
        return (self.scale * x) + self.bias

class ElementwiseNonlin(nn.Module):
    """
    Captures the number of features in a nonlinearity, so KLD can learn a separate
    linear approximation to each feature.
    """
    def __init__(self, mod_or_func, features):
        super().__init__()
        self.mod_or_func = mod_or_func
        self.features = features

    def forward(self, x):
        assert x.shape[-1] == self.features
        return self.mod_or_func(x)

#############################
#### Reshaping functions ####
#############################

class MethodFunction(nn.Module):
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

class TupleArgFunction(MethodFunction):
    def __call__(self, xs):
        validate_tuple(xs)
        return self.function(xs, *self.args, **self.kwargs)


restructure_methods = {
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
    'unbind': 'UnBind',
}

for method_name, class_name in restructure_methods.items():
    new_class = type(class_name, (Method,), {
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
    new_class = type(class_name, (TupleArgFunction,), {
        'function': function
    })
    globals()[class_name] = new_class
