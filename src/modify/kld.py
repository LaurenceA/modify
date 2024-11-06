import torch
import torch.nn as nn

import modify

##########################################
#### Type for holding Gamma and gamma ####
##########################################

class Gamma():
    """
    G is a matrix, \Gamma = <dL/dx x^T>
    g is a vector, \gamma = <dL/dx>
    G.shape = [dim, dim]
    g.shape = [dim, 1]
    either G or g may be None.
    """
    def __init__(self, G, g):
        assert isinstance(G, (torch.Tensor, type(None)))
        assert isinstance(g, (torch.Tensor, type(None)))

        if G is not None:
            assert isinstance(G, torch.Tensor)
            assert 2 == G.ndim
            assert G.shape[0] == G.shape[1]

        if g is not None:
            assert isinstance(g, torch.Tensor)
            assert 2 == g.ndim
            assert 1 == g.shape[1]

        if (G is not None) and (g is not None):
            assert G.shape[0] == g.shape[0]

        self.G = G
        self.g = g

def validate_tuple(xs):
    """
    Checks that the input is a tuple containing (possibly nested tuples of) Gamma's.
    """
    assert isinstance(xs, tuple)
    for x in xs:
        validate_tuple_or_gamma(x)

def validate_tuple_or_gamma(xs):
    """
    Checks that the input is a Gamma or a (nested) tuple of Gamma's.
    """
    assert isinstance(xs, (tuple, Gamma))
    if isinstance(xs, tuple):
        for x in xs:
            validate_tuple_or_gamma(x)

#################################
#### Parallel and Sequential ####
#################################

class ModuleGroup(modify.ModuleGroup):
    def chol_vec(self, module_inputs):
        result, gamma_out = self.mat_vec_prod('_chol_vec', module_inputs, Gamma(None, None))
        return result

    def inv_chol_vec(self, module_inputs):
        result, gamma_out = self.mat_vec_prod('_inv_chol_vec', module_inputs, Gamma(None, None))
        return result

    def _chol_vec(self, module_inputs, gamma_in=Gamma(None, None)):
        return self.mat_vec_prod('_chol_vec', module_inputs, gamma_in)

    def _inv_chol_vec(self, module_inputs, gamma_in=Gamma(None, None)):
        return self.mat_vec_prod('_inv_chol_vec', module_inputs, gamma_in)

    def log_det_inv_chol(mod):
        """
        Returns the log |chol^{-1}(cov)}|
        """
        terms = [mod.log_det_inv_chol() for mod in mod.mods.values()]
        return sum_none(*terms)

class Parallel(ModuleGroup):
    def mat_vec_prod(self, method, module_inputs, gamma_ins):
        assert isinstance(module_inputs, dict)
        validate_tuple(gamma_ins)

        module_results = {}
        gamma_outs = []
        for ((name, mod), gamma_in) in zip(self.mods.items(), gamma_ins):
            module_result, gamma_out = getattr(mod, method)(module_inputs[name], gamma_in)
            module_results[name] = module_result
            gamma_outs.append(gamma_out)
        return (module_results, gamma_outs)
        
class Sequential(ModuleGroup):
    def mat_vec_prod(self, method, module_inputs, gamma):
        assert isinstance(module_inputs, dict)
        validate_tuple_or_gamma(gamma)

        module_results = {}
        for name, mod in self.mods.items():
            module_result, gamma = getattr(mod, method)(module_inputs[name], gamma)
            module_results[name] = module_result
        return (module_results, gamma)

##########################
#### No Param Modules ####
##########################

class NoParamModule(nn.Module):
    """
    For stuff like nonlinearities, Copy, Add, RMSNorm, LayerNorm.
    All these modules don't have any parameters, so don't have any
    gradients, and hence no module_inputs or module_results.  
    All we need to do is propagate the Gamma.

    This is an abstract class; concrete classes must override gamma_out.
    """
    def check_inputs(self, module_inputs, gamma):
        assert isinstance(gamma, Gamma)
        assert isinstance(module_inputs, dict) and (0==len(module_inputs))

    def _inv_chol_vec(self, module_inputs, gamma_in):
        self.check_inputs(module_inputs, gamma_in)
        return ({}, self.gamma_out(gamma_in))

    def _chol_vec(self, module_inputs, gamma_in):
        self.check_inputs(module_inputs, gamma_in)
        return ({}, self.gamma_out(gamma_in))

    def log_det_inv_chol(self):
        return None

def prod_none(*xs):
    """
    returns None if either of the inputs is None.
    """
    if any(x is None for x in xs):
        return None
    else:
        total = xs[0]
        for x in xs[1:]:
            total = total * x
        return total

def matmul_none(x, y):
    """
    x@y, returns None if either of the inputs is None.
    """
    if (x is not None) and (y is not None):
        return x@y
    else:
        return None

def sum_none(*xs):
    """
    Returns sum of all non-None inputs.  Returns None if all inputs are None.
    """
    non_none_xs = [x for x in xs if x is not None]
    if 0 == len(non_none_xs):
        return None
    else:
        total = non_none_xs[0]
        for x in non_none_xs[1:]:
            total = total + x
        return total

class ElementwiseNonlin(NoParamModule):
    def __init__(self, mod, indep_across_layers):
        super().__init__()
        assert isinstance(mod, modify.ElementwiseNonlin)
        self.a = nn.Parameter(torch.ones(mod.features))
        self.b = nn.Parameter(torch.zeros(mod.features))
        self.s = nn.Parameter(torch.ones(mod.features, 1))

    def gamma_out(self, gamma_in):
        G = prod_none(
            self.s, 
            sum_none(
                prod_none(self.a, gamma_in.G), 
                prod_none(self.b, gamma_in.g)
            )
        )
        g = prod_none(self.s, gamma_in.g)
        return Gamma(G, g)
        

class SPDA(NoParamModule):
    def __init__(self, mod, indep_across_layers):
        super().__init__()

    def gamma_out(self, gamma_ins):
        validate_tuple(gamma_ins)
        #Just return gammas from the values.
        return gamma_in[2]

class Copy(NoParamModule):
    def __init__(self, mod, indep_across_layers):
        super().__init__()
        self.number_of_copies = mod.number_of_copies

    def gamma_out(self, gamma_in):
        return self.number_of_copies * (gamma_in,)

class Add(NoParamModule):
    def __init__(self, mod, indep_across_layers):
        super().__init__()
        self.omega1_G = nn.Parameter(0.5*torch.ones(()))
        self.omega2_G = nn.Parameter(0.5*torch.ones(()))
        self.omega1_g = nn.Parameter(0.5*torch.ones(()))
        self.omega2_g = nn.Parameter(0.5*torch.ones(()))

    def gamma_out(self, gamma_ins):
        assert 2 == len(gamma_ins)
        G = sum_none(
            prod_none(self.omega1_G, gamma_ins[0].G),
            prod_none(self.omega2_G, gamma_ins[1].G),
        )
        g = sum_none(
            prod_none(self.omega1_g, gamma_ins[0].g),
            prod_none(self.omega2_g, gamma_ins[1].g),
        )
        
        return Gamma(G, g)

class Mul(NoParamModule):
    def __init__(self, mod, indep_across_layers):
        super().__init__()
        self.s0 = nn.Parameter(torch.ones((mod.features, 1)))
        self.s1 = nn.Parameter(torch.ones((mod.features, 1)))
        self.a0 = nn.Parameter(torch.ones(mod.features))
        self.a1 = nn.Parameter(torch.ones(mod.features))
        self.b  = nn.Parameter(torch.zeros(mod.features))

    def gamma_out(self, gamma_ins):
        assert 2 == len(gamma_ins)
        g = sum_none(
            prod_none(self.s0, gamma_ins[0].g),
            prod_none(self.s1, gamma_ins[1].g),
        )
        G = sum_none(
            prod_none(gamma_ins[0].G, self.s0, self.a0),
            prod_none(gamma_ins[1].G, self.s1, self.a1),
            prod_none(gamma_ins[0].g, self.s0, self.b),
            prod_none(gamma_ins[1].g, self.s1, self.b),
        )
        
        return Gamma(G, g)

class MeanSub(NoParamModule):
    def __init__(self, mod, indep_across_layers):
        super().__init__()

    def gamma_out(self, gamma_in):
        g = gamma_in.g
        if g is not None:
            g = g - g.mean(1, keepdim=True)

        G = gamma_in.G
        if G is not None:
            G = G - gamma_in.G.mean(0, keepdim=True)
            G = G - gamma_in.G.mean(1, keepdim=True)
        
        return Gamma(G, g)

class RMSNorm(NoParamModule):
    def __init__(self, mod, indep_across_layers):
        self.log_eps  = nn.Parameter(-10 * torch.ones(()))
        self.xb = nn.Parameter(torch.randn(mod.normalized_shape[-1], 1))

    def gamma_out(self, gamma_in):
        xb = self.xb
        norm = xb.mT@xb + self.log_eps.exp()

        g = gamma_in.g
        if g is not None:
            g = torch.sqrt(norm) * (g - xb @ ((xb.mT @ g) / norm))

        G = gamma_in.G
        if G is not None:
            G = G - xb @ ((xb.mT @ G) / norm)
            G = G - ((G @ xb) / norm) @ xb.mT
        
        return Gamma(G, g)


#################################
#### Classes with parameters ####
#################################

class PositiveTriangular(nn.Module):
    """
    A triangular matrix, with all the diagonal elements being positive.
    """
    def __init__(self, features, upper):
        super().__init__()
        assert isinstance(features, int)
        assert isinstance(upper, bool)
        self.features = features
        self.upper = upper
        self.A = nn.Parameter(torch.zeros(features, features))
        self.log_diag = nn.Parameter(torch.zeros(features))

    def forward(self):
        if self.upper:
            off_diag = self.A.triu(1)
        else:
            off_diag = self.A.tril(-1)

        return self.log_diag.exp().diag() + off_diag

    def logdet(self):
        return self.log_diag.sum()


class Linear(nn.Module):
    def __init__(self, mod, indep_across_layers):
        super().__init__()
        assert isinstance(mod, nn.Linear)
        self.indep_across_layers = indep_across_layers

        #Save weight matrix and bias vector for the underlying linear module.
        #The .data takes mod.weight, which is an nn.Parameter and returns the underlying Tensor.
        #this tensor won't e.g. appear in grad_model.parameters().
        self.register_buffer('weight', mod.weight.data)                           # features
        self.register_buffer('bias', None if mod.bias is None else mod.bias.data) # features or None

        self.out_features, self.in_features = self.weight.shape
        self.in_features_bias = self.in_features + int(mod.bias is not None)
        if self.bias is not None:
            self.bias.shape == (self.out_features,)

        #Approximation to the inverse weight matrix for inverting backprop.
        self.inv_weight_T = nn.Parameter(torch.zeros(mod.weight.shape)) # out_features x in_features

        #Inverses of the Kronecker factored Cholesky of the covariance matrix.

        #chol^{-1}(cov_left), i.e. lower-triangular matrix, representing the 
        #inverse of the Cholesky of the left Kronecker factor of the covariance.
        self.invL = PositiveTriangular(self.out_features,     upper=False)
        #chol^{-T}(cov_right) i.e. upper-triangular matrix, representing the
        #transpose of the inverse of the Cholesky of the right Kronecker factor of the covariance.
        self.invU = PositiveTriangular(self.in_features_bias, upper=True)

    def log_det_inv_chol(self):
        return self.in_features_bias * self.invL.logdet() + self.out_features * self.invU.logdet()

    def pred_weight_grad(self, gamma_in):
        #gamma.G         is  in_features x in_features
        #self.weight_inv is out_features x in_features
        if self.indep_across_layers:
            return None
        else:
            return matmul_none(self.inv_weight_T, gamma_in.G) # out_features x in_features.

    def pred_bias_grad(self, gamma_in):
        #gamma.g         is  in_features x 1
        #self.weight_inv is out_features x in_features
        if self.indep_across_layers:
            return None
        else:
            return matmul_none(self.inv_weight_T, gamma_in.g) # out_features x 1.

    def check(self, module_inputs, gamma_in):
        assert isinstance(gamma_in, Gamma)
        if gamma_in.G is not None:
            assert gamma_in.G.shape == (self.in_features, self.in_features)
        if gamma_in.g is not None:
            assert gamma_in.g.shape == (self.in_features, 1)

        assert isinstance(module_inputs, dict)
        assert module_inputs["weight"].shape == (self.out_features, self.in_features)

        #Check gradients / noise in module_inputs match the shape of the weight and bias params.
        assert (self.bias is not None) == ('bias' in module_inputs)
        if 'bias' in module_inputs:
            assert module_inputs["bias"].shape == (self.out_features,)

    def gamma_out(self, grads, gamma_in):
        G = grads['weight'] @ self.weight.mT
        if 'bias' in grads:
            g = grads['bias'][:, None]
        elif gamma_in.g is not None:
            g = self.inv_weight_T @ gamma_in.g
        else:
            g = None
        return Gamma(G, g)


    def _inv_chol_vec(self, grad_module_inputs, gamma_in):
        """
        Converts gradients sampled from the gradient model into IID noise.
        """
        self.check(grad_module_inputs, gamma_in)

        grad      = grad_module_inputs['weight']    # out_features x in_features
        pred_grad = self.pred_weight_grad(gamma_in) # out_features x in_features or None
        if pred_grad is None:
            pred_grad = torch.zeros_like(grad)          # out_features x in_features

        if self.bias is not None:
            grad_bias = grad_module_inputs['bias'][:, None]        # out_features x 1
            pred_grad_bias = self.pred_bias_grad(gamma_in)         # out_features x 1 or None
            if pred_grad_bias is None:
                pred_grad_bias = torch.zeros_like(grad_bias)                # out_features x in_features
            grad      = torch.cat((     grad,      grad_bias), -1) # out_features x in_features+1
            pred_grad = torch.cat((pred_grad, pred_grad_bias), -1) # out_features x in_features+1 or None

        corr_noise = grad - pred_grad                       # out_features x in_features+1?
        iid_noise  = self.invL() @ corr_noise @ self.invU() # out_features x in_features+1?

        if self.bias is not None:
            noise_module_outputs = {'weight': iid_noise[:, :-1], 'bias': iid_noise[:, -1]}
        else:
            noise_module_outputs = {'weight': iid_noise}

        return (noise_module_outputs, self.gamma_out(grad_module_inputs, gamma_in))

    def _chol_vec(self, noise_module_inputs, gamma_in):
        """
        Converts IID noise into gradients from the gradient model.
        """
        self.check(noise_module_inputs, gamma_in)

        iid_noise = noise_module_inputs['weight']   # out_features x in_features
        pred_grad = self.pred_weight_grad(gamma_in) # out_features x in_features or None
        if pred_grad is None:
            pred_grad = torch.zeros_like(iid_noise)     # out_features x in_features

        if self.bias is not None:
            iid_noise_bias = noise_module_inputs['bias'][:, None]    # out_features x 1
            pred_grad_bias = self.pred_bias_grad(gamma_in)           # out_features x 1 or None
            if pred_grad_bias is None:
                pred_grad_bias = torch.zeros_like(iid_noise_bias)             # out_features x 1
            iid_noise = torch.cat((iid_noise, iid_noise_bias), -1) # out_features x in_features+1
            pred_grad = torch.cat((pred_grad, pred_grad_bias), -1) # out_features x in_features+1

        #Computes:
        #corr_noise = L^{-1} @ corr_noise @ U^{-1}
        corr_noise = torch.linalg.solve_triangular(
            self.invL(), 
            torch.linalg.solve_triangular(self.invU(), iid_noise, upper=True, left=False),
            upper=False,
        )                             # out_features x in_features+1?
        grad = corr_noise + pred_grad # out_features x in_features+1?

        if self.bias is not None:
            grad_module_outputs = {'weight': grad[:, :-1], 'bias': grad[:, -1]}
        else:
            grad_module_outputs = {'weight': grad}

        return (grad_module_outputs, self.gamma_out(grad_module_outputs, gamma_in))
    

class ElementwiseAffine():
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
    def __init__(self, mod, indep_across_layers):
        super().__init__()
        self.indep_across_layers = indep_across_layers
        assert isinstance(mod, modify.ElementwiseAffine)

        #Save weight matrix and bias vector for the underlying linear module.
        #The .data takes mod.weight, which is an nn.Parameter and returns the underlying Tensor.
        #this tensor won't e.g. appear in grad_model.parameters().
        self.register_buffer('weight', mod.weight.data)                           # features
        self.register_buffer('bias', None if mod.bias is None else mod.bias.data) # features or None

        self.features = self.a.shape[0]
        assert self.a.shape == (features,)

        self.log_inv_std_weight = nn.Parameter(torch.ones(features))

        if self.bias is not None:
            assert self.bias.shape == (features,)
            self.log_inv_std_bias = nn.Parameter(torch.ones(features))

        self.s = nn.Parameter(torch.ones(self.features, 1))

    def log_det_inv_chol(self):
        result = self.log_inv_std_weight.sum()
        if self.bias is not None:
            result = result + self.log_inv_std_bias.sum()
        return result

    def pred_weight_grad(self, gamma_in):
        if gamma_in.G is not None and not self.indep_across_layers:
            return self.s*gamma_in.G.diagonal()
        else:
            None

    def pred_bias_grad(self, gamma_in):
        if not self.indep_across_layers:
            return mul_none(self.s, gamma_in.g)
        else:
            return None

    def check(self, gamma_in, module_inputs):
        assert isinstance(gamma_in, Gamma)
        if gamma_in.G is not None:
            gamma_in.G.shape == (self.features, self.features)

        if gamma_in.g is not None:
            gamma_in.g.shape == (self.features,)
        
        assert isinstance(module_inputs, dict)
        module_inputs['weight'].shape == (self.features,)
        if 'bias' in module_inputs:
            module_inputs['bias'].shape == (self.features,)

    def gamma_out(self, grads, gamma_in):
        G = sum_none(
            prod_none(gamma_in.G, self.s, self.weight),
            prod_none(gamma_in.g, self.s, self.bias)
        )

        if 'bias' in grads:
            g = grads['bias']
        elif gamma_in.g is not None:
            g = self.s * gamma_in.g
        else:
            g = None
        return Gamma(G, g)


    def _inv_chol_vec(self, grad_module_inputs, gamma_in):
        """
        Converts gradients sampled from the gradient model into IID noise.
        """
        self.check(grad_module_inputs, gamma_in)

        noise_module_outputs = {}

        grad_weight       = grad_module_inputs['weight']
        corr_noise_weight = grad - self.pred_weight_grad(gamma_in)
        iid_noise_weight  = self.log_inv_std_weight.exp() * corr_noise_weight 
        noise_module_outputs['weight'] = iid_noise_weight 

        if self.bias is not None:
            grad_bias       = grad_module_inputs['bias']
            corr_noise_bias = grad - self.pred_bias_grad(gamma_in)
            iid_noise_bias  = self.log_inv_std_bias.exp() * corr_noise_bias 
            noise_module_outputs['bias'] = iid_noise_bias 

        return (noise_module_outputs, self.gamma_out(grad_module_inputs, gamma_in))

    def _chol_vec(self, noise_module_inputs, gamma_in):
        """
        Converts IID noise into gradients from the gradient model.
        """
        self.check(noise_module_inputs, gamma_in)

        grad_module_outputs = {}

        iid_noise_weight  = noise_module_inputs['weight']
        corr_noise_weight = (-self.log_inv_std_weight).exp() * iid_noise_weight
        grad_weight       = corr_noise_weight + self.pred_weight_grad(gamma_in)
        grad_module_outputs['weight'] = grad

        if self.bias is not None:
            iid_noise_bias  = noise_module_inputs['bias']
            corr_noise_bias = (-self.log_inv_std_bias).exp() * iid_noise_bias
            grad_bias       = corr_noise_bias + self.pred_bias_grad(gamma_in)
            grad_module_outputs['bias'] = grad_bias

        return (grad_module_outputs, self.gamma_out(gamma_in, grad_module_outputs))

def rms_norm(mod):
    result = [RMSNorm(mod)]
    if mod.affine:
        result.append(ElementwiseAffine(mod))
    return modify.Sequential(result)

def layer_norm(mod):
    result = [MeanSub(mod), RMSNorm(mod)]
    if mod.affine:
        result.append(ElementwiseAffine(mod))
    return modify.Sequential(result)

kld_module_group = {
    modify.Sequential: Sequential,
    modify.Parallel: Parallel,
}
kld_modules = {
    modify.ElementwiseNonlin: ElementwiseNonlin,
    nn.Linear: Linear,
    modify.ElementwiseAffine: ElementwiseAffine,
    modify.Copy: Copy,
    modify.Add: Add,
    modify.Mul: Mul,
    nn.RMSNorm: rms_norm,
    nn.LayerNorm: layer_norm,
}

def kldify(mod, indep_across_layers=False):
    if isinstance(mod, modify.ModuleGroup):
        return kld_module_group[type(mod)]({k: kldify(v, indep_across_layers) for (k, v) in mod.mods.items()})
    elif type(mod) in kld_modules:
        return kld_modules[type(mod)](mod, indep_across_layers)
    else:
        raise Exception(f"KLD doesn't know how to handle {type(mod)}")

def param2dict(f, mod):
    """
    Applies a function, f, to every parameter, and returns a nested dict of the results.
    """
    result = {}
    if isinstance(mod, modify.ModuleGroup):
        for (k, m) in mod.mods.items():
            result[k] = param2dict(f, m)
    else:
        for (k, v) in mod.named_parameters():
            result[k] = f(v)
    return result

def grad2dict(mod):
    """
    Takes a model with gradients on the parameters, and converts to a nested dict.
    """
    def f(v):
        assert v.grad is not None
        return v.grad

    return param2dict(f, mod)
    #result = {}
    #if isinstance(mod, modify.ModuleGroup):
    #    for (k, m) in mod.mods.items():
    #        result[k] = grad2dict(m)
    #else:
    #    for (k, v) in mod.named_parameters():
    #        assert v.grad is not None
    #        result[k] = v.grad
    #return result

def noise_dict(mod):
    """
    Takes a model without gradients on the parameters, and returns a nested dict with IID
    noise the same shape as the parameters
    """
    return param2dict(lambda v: torch.randn_like(v), mod)

def rms_grad(mod):
    return param2dict(lambda v: v.grad.square().mean().sqrt().item(), mod)

def dict2grad(grad_dict, mod):
    """
    Takes a model with gradients on the parameters, and converts to a nested dict.
    """
    assert isinstance(grad_dict, dict)

    if isinstance(mod, modify.ModuleGroup):
        for (k, m) in mod.mods.items():
            dict2grad(grad_dict[k], m)
    else:
        for (k, v) in mod.named_parameters():
            v.grad = grad_dict[k]

def map_dict(f, d):
    result = {}
    for k, v in result.items():
        if isinstance(v, dict):
            result[k] = map_dict(f, v)
        else:
            result[k] = f(v)
    return result

def square_sum(d):
    terms = [square_sum(v) if isinstance(v, dict) else v.square().sum() for v in d.values()]
    return sum_none(*terms)

def grad_model_loss(model, grad_model):
    """
    Computes log probability of a multivariate Gaussian distribution over the gradients.
    """
    grad_dict = grad2dict(model)
    noise_dict = grad_model.inv_chol_vec(grad_dict)
    log_prob = -0.5*square_sum(noise_dict) + grad_model.log_det_inv_chol()
    return -log_prob
    
def natural_grad(model, grad_model):
    """
    Applies a natural gradient update using the gradient model.
    """
    #Extract .grad from the model parameters and puts them in a dict
    grad_dict = grad2dict(model)

    #noise_values = xi = L^{-1} grad
    noise_dict, vjp_fn = torch.func.vjp(grad_model.inv_chol_vec, grad_dict)

    #Computes natgrad = L^{-T} xi = L^{-T} L^{-1} grad
    natgrad_dict = vjp_fn(noise_dict)[0] #[0] is for the outer tuple.

    #Puts the natural gradient updates back into .grad, ready to be used with a standard optimizer.
    dict2grad(natgrad_dict, model)

def sample_grad_model(model, grad_model):
    """
    Samples some gradients that should look like gradients from the real model
    with sampled data.
    """
    iid_noise_dict = noise_dict(model)

    grad_dict = grad_model.chol_vec(iid_noise_dict)

    #Puts the natural gradient updates back into .grad, ready to be used with a standard optimizer.
    dict2grad(grad_dict, model)
