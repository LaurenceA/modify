import torch
import torch.nn as nn
import modify

class ModuleGroup(modify.ModuleGroup):
    def chol_vec(self, Gamma, input_vec):
        return self.mat_vec_prod(Gamma, input_vec, 'chol_vec'):

    def inv_chol_vec(self, Gamma, input_vec):
        return self.mat_vec_prod(Gamma, input_vec, 'inv_chol_vec'):

class Parallel(modify.ModuleGroup):
    def mat_vec_prod(self, Gammas, input_vec, method):
        modify.validate_tuple(Gammas)

        result_vec = {}
        result_Gammas = []
        for name, mod in self.mods.items():
            _result_Gammas, _result_vec = getattr(mod, method)(Gammas, input_vec[name])
            result_vec[name] = _result_vec
            result_Gammas.append(_result_Gammas)
        return result_Gammas, result_vec
        
class Sequential(modify.ModuleGroup):
    def mat_vec_prod(self, Gammas, input_vec, method):
        modify.validate_tuple_or_tensor(Gammas)
        result_vec = {}
        for name, mod in self.mods.items():
            Gammas, _result_vec = getattr(mod, method)(Gammas, input_vec[name])
            result_vec[name] = _result_vec
        return Gammas, result_vec

class NoParamModule(nn.Module):
    """
    For stuff like nonlinearities, Copy, Add, RMSNorm, LayerNorm.
    All these modules don't have any parameters, so don't have any
    gradients.  All we need to do is propagate the Gamma.
    """
    def check_inputs(self, Gamma, input_vec)
        assert isinstance(Gamma, torch.tensor)
        assert isinstance(input_vec, dict) and (0==len(input_vec))

    def inv_chol_vec(self, Gamma, input_vec):
        self.check_inputs(Gamma, input_vec)
        return (self._forward(Gamma), {})

    def chol_vec(self, Gamma, input_vec):
        self.check_inputs(Gamma, input_vec)
        return (self._forward(Gamma), {})

class Pointwise(NoParamModule):
    def __init__(self, mod):
        assert isinstance(mod, pointwise_nonlin)
        a = nn.Parameter(t.ones(1, mod.features))
        s = nn.Parameter(t.ones(mod.features, 1))

    def _forward(self, Gamma):
        return (Gamma*a)*s

class ElementwiseAffine()
    """
    ElementwiseAffine corresponds to a bias + scale (e.g. after LayerNorm).
    Of course, we do have gradients for the bias + scale parameters, but
    they're not very useful for predicting future gradients because they're
    low-dimensional (just features, as opposed to in_features x out_features
    for weights).

    We therefore use the input Gamma to predict the gradients of bias + scale,
    but we don't bother to use the gradients of bias + scale to update output
    Gamma.
    """
    def __init__(self, mod):
        assert isinstance(mod, modify.ElementwiseAffine)
        self.mod = [mod]

    def inv_chol_vec(self, Gamma, g):
        self.check_inputs(Gamma, input_vec)
        return (self._forward(Gamma), {})

    def chol_vec(self, Gamma, xi):
        self.check_inputs(Gamma, input_vec)
        return (self._forward(Gamma), {})


kld_module_group = {
    modify.Sequential: Sequential,
    modify.Parallel: Parallel,
}
kld_modules = {
    modify.Pointwise: Pointwise
    modify.Linear: Linear,
    modify.ElementWiseAffine: ElementWiseAffine,
    modify.Copy: Copy,
    modify.Add: Add,
    nn.RMSNorm: RMSNorm,
    nn.LayerNorm: LayerNorm,
}

def kldify(mod):
    if isinstance(mod, modify.ModuleGroup):
        return kld_module_group[type(mod)]({k: kldify(v) for (k, v) in mod.mods})
    elif type(mod) in kld_modules:
        return kld_modules[type(mod)](mod)
    else:
        raise Exception(f"KLD doesn't know how to handle {type(mod)}")
        
        
