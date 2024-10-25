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


class ElementwiseAffine(_ModifyModule,_Unary,_Elementwise,_Param,_Linear):
    """
    Does elementwise multiplication and biasing (e.g. usually used after norm layers)
    """
    def __init__(self, shape, bias=True):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.weights = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        result = self.weights * x
        if self.bias is not None:
            result = result + bias
        return result


TorchLearnedLinear = (
    nn.Linear, 
    nn.Conv1d, 
    nn.Conv2d, 
    nn.Conv3d, 
    nn.ConvTranspose1d, 
    nn.ConvTranspose2d, 
    nn.ConvTranspose3d,
    nn.Embedding,
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
