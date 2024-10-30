import torch
import torch.nn as nn

import modify

#################################
#### Type for holding Gammas ####
#################################

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
            assert G.shape[1] == g.shape[1]

        self.G = G
        self.g = g

def validate_tuple(xs):
    """
    Checks that the input is a tuple containing (possibly nested tuples of) Gammas.
    """
    assert isinstance(xs, tuple)
    for x in xs:
        validate_tuple_or_gammas(x)

def validate_tuple_or_gamma(xs):
    """
    Checks that the input is a Gamma or a (nested) tuple of Gammas.
    """
    assert isinstance(xs, (tuple, Gamma))
    for x in xs:
        validate_tuple_or_gammas(x)

#################################
#### Parallel and Sequential ####
#################################

class ModuleGroup(modify.ModuleGroup):
    def chol_vec(self, gamma_in, module_inputs):
        return self.mat_vec_prod(gamma_in, module_inputs, 'chol_vec')

    def inv_chol_vec(self, gamma_in, module_inputs):
        return self.mat_vec_prod(gamma_in, module_inputs, 'inv_chol_vec')

class Parallel(modify.ModuleGroup):
    def mat_vec_prod(self, gamma_ins, module_inputs, method):
        validate_tuple(gamma_ins)

        module_results = {}
        gamma_outs = []
        for name, mod in self.mods.items():
            gamma_out, module_result = getattr(mod, method)(Gammas, module_inputs[name])
            module_results[name] = module_result
            gamma_outs.append(gamma_out)
        return gamma_outs, module_results
        
class Sequential(modify.ModuleGroup):
    def mat_vec_prod(self, gamma, module_inputs, method):
        validate_tuple_or_gammas(gamma)
        module_results = {}
        for name, mod in self.mods.items():
            gamma, _module_results = getattr(mod, method)(gamma, module_inputs[name])
            module_results[name] = _module_results
        return gamma, module_results

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
    def check_inputs(self, Gamma, module_inputs):
        assert isinstance(Gamma, torch.tensor)
        assert isinstance(module_inputs, dict) and (0==len(module_inputs))

    def inv_chol_vec(self, gamma_in, module_inputs):
        self.check_inputs(gamma_in, module_inputs)
        return (self.gamma_out(gamma_in), {})

    def chol_vec(self, gamma_in, module_inputs):
        self.check_inputs(gamma_in, module_inputs)
        return (self._forward(gamma_in), {})

def prod_none(*xs):
    """
    returns None if either of the inputs is None.
    """
    if any(x is None in xs):
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
    non_none_xs = [x for x in xs if x is not None]
    if 0 == len(non_non_xs):
        return None
    else:
        total = non_none_xs[0]
        for x in non_none_xs[1:]:
            total = total + x
        return total

class ElementwiseNonlin(NoParamModule):
    def __init__(self, mod):
        super().__init__()
        assert isinstance(mod, modify.ElementwiseNonlin)
        self.a = nn.Parameter(torch.ones(mod.features))
        self.b = nn.Parameter(torch.zeros(mod.features))
        self.s = nn.Parameter(torch.ones(mod.features, 1))

    def gamma_out(self, gamma_in):
        G = mul_none(
            self.s, 
            sum_none(
                prod_none(self.a, gamma_in.G), 
                prod_none(self.b, gamma_in.g)
            )
        )
        g = prod_none(self.s, gamma_in.g)
        return Gammas(G, g)
        

class SPDA(NoParamModule):
    def __init__(self, mod):
        super().__init__()

    def gamma_out(self, gamma_ins):
        validate_tuple(gamma_ins)
        #Just return gammas from the values.
        return gamma_in[2]

class Copy(NoParamModule):
    def __init__(self, mod):
        super().__init__()
        self.number_of_copies = mod.number_of_copies

    def gamma_out(self, gamma_in):
        return self.number_of_copies * (gamma_in,)

class Add(NoParamModule):
    def __init__(self, mod):
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
        
        return Gammas(G, g)

class Mul(NoParamModule):
    def __init__(self, mod):
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
        
        return Gammas(G, g)

class MeanSub(NoParamModule):
    def gamma_out(self, gamma_in):
        g = gamma_in.g
        if g is not None:
            g = g - g.mean(1, keepdim=True)

        G = gamma_in.G
        if G is not None:
            G = G - gamma_in.G.mean(0, keepdim=True)
            G = G - gamma_in.G.mean(1, keepdim=True)
        
        return Gammas(G, g)

class RMSNorm(NoParamModule):
    def __init__(self, mod):
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
        
        return Gammas(G, g)


#################################
#### Classes with parameters ####
#################################

class PositiveTriangular(nn.Module):
    """
    A triangular matrix, with all the diagonal elements being positive.
    """
    def __init__(self, features, upper):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(features, features))
        self.log_diag = nn.Parameter(torch.zeros(features))

    def forward(self):
        if upper:
            off_diag = self.A.triu(1)
        else:
            off_diag = self.A.tril(-1)

        return self.log_diag.exp().diag() + off_diag

    def logdet(self):
        return self.log_diag.sum()


class Linear(nn.Module):
    def __init__(self, mod):
        super().__init__()
        assert isinstance(mod, nn.Linear)

        #Save weight matrix and bias vector for the underlying linear module.
        self.weight = mod.weight # out_features x in_features
        self.bias = mod.bias     # None or out_features

        self.out_features, self.in_features = self.weight.shape
        self.in_features_bias = self.in_features +  + int(mod.bias is None)
        if self.bias is not None:
            self.bias.shape == (self.out_features,)

        #Approximation to the inverse weight matrix for inverting backprop.
        self.inv_weight_T = nn.Parameter(torch.zeros(mod.weight.shape)) # out_features x in_features

        #Inverse of the Kronecker factored Cholesky; do the trick of treating the bias as an extra feature.
        self._invL = PositiveTriangular(self.out_features,     upper=False)
        self._invU = PositiveTriangular(self.in_features_bias, upper=True)

    @property
    def invL(self):
        """
        chol^{-1}(cov_left), i.e. lower-triangular matrix, representing the 
        inverse of the Cholesky of the left Kronecker factor of the covariance.
        """
        return self._invL.tril()

    @property
    def invU(self):
        """
        chol^{-T}(cov_right) i.e. upper-triangular matrix, representing the
        transpose of the inverse of the Cholesky of the right Kronecker factor of the covariance.
        """
        return self._invU.triu()

    def pred_grad_weight(self, gamma_in):
        #gamma.G         is  in_features x in_features
        #self.weight_inv is out_features x in_features
        return matmul_none(self.inv_weight_T, gamma_in.G) # out_features x in_features.

    def pred_grad_bias(self, gamma_in):
        #gamma.g         is  in_features x 1
        #self.weight_inv is out_features x in_features
        return matmul_none(self.inv_weight_T, gamma_in.g) # out_features x 1.

    def check(self, gamma_in, module_inputs):
        if gamma_in.G is not None:
            assert gamma_in.G.shape == (self.in_features, self.in_features)
        if gamma_in.g is not None:
            assert gamma_in.G.shape == (self.in_features, 1)

        assert module_inputs["weight"].shape == (self.out_features, self.in_features)

        #Check gradients / noise in module_inputs match the shape of the weight and bias params.
        assert (self.bias is not None) == ('bias' in module_inputs)
        if 'bias' in module_inputs:
            assert module_inputs["bias"].shape == (self.out_features,)

    def gamma_out(self, gamma_in, grads):
        G = grads['weight'] @ self.weight.mT
        if 'bias' in grads:
            g = grads['bias']
        elif gamma_in.g is not None:
            g = self.inv_weight_T @ gamma_in.g
        else:
            g = None
        return Gamma(G, g)


    def inv_chol_vec(self, gamma_in, grad_module_inputs):
        """
        Converts gradients sampled from the gradient model into IID noise.
        """
        self.check(gamma_in, grad_module_inputs)

        grad      = grad_module_inputs['weight']          # out_features x in_features
        pred_grad = self.pred_weight_grad(self, gamma_in) # out_features x in_features

        if self.bias is not None:
            grad_bias = grad_module_inputs['bias']               # out_features x 1
            pred_grad_bias = self.pred_bias_grad(self, gamma_in) # out_features x 1
            grad      = torch.cat((     grad,      grad_bias), -1)   # out_features x in_features+1
            pred_grad = torch.cat((pred_grad, pred_grad_bias), -1)   # out_features x in_features+1

        corr_noise = grad - pred_grad             # out_features x in_features+1?
        iid_noise  = self.L @ corr_noise @ self.U # out_features x in_features+1?

        if self.bias is not None:
            noise_module_outputs = {'bias': iid_noise[:, -1:], 'weight': iid_noise[:, :-1]}
        else:
            noise_module_outputs = {'weight': iid_noise}

        return (self.gamma_out(gamma_in, grad_module_inputs), noise_module_outputs)

    def chol_vec(self, gamma_in, noise_module_inputs):
        """
        Converts IID noise into gradients from the gradient model.
        """
        self.check(gamma_in, noise_module_inputs)

        iid_noise = noise_module_inputs['weight']      # out_features x in_features
        pred_grad = self.pred_weight_grad(self, gamma_in) # out_features x in_features

        if self.bias is not None:
            iid_noise_bias = noise_module_inputs['bias']         # out_features x 1
            pred_grad_bias = self.pred_bias_grad(self, gamma_in) # out_features x 1
            iid_noise = torch.cat(( iid_noise,  iid_noise_bias), -1) # out_features x in_features+1
            pred_grad = torch.cat((pred_noise, pred_noise_bias), -1) # out_features x in_features+1

        #Computes:
        #corr_noise = L^{-1} @ corr_noise @ U^{-1}
        corr_noise = torch.linalg.solve_triangular(
            self.L, 
            torch.linalg.solve_triangular(self.U, iid_noise, upper=True, left=False),
            upper=False,
        )                             # out_features x in_features+1?
        grad = corr_noise + pred_grad # out_features x in_features+1?

        if self.bias is not None:
            grad_module_outputs = {'bias': grad[:, -1:], 'weight': grad[:, :-1]}
        else:
            grad_module_outputs = {'weight': grad}

        return (self.gamma_out(gamma_in, grad_module_outputs), grad_module_outputs)
    

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
    def __init__(self, mod):
        assert isinstance(mod, modify.ElementwiseAffine)
        self.weight = mod.weight
        self.bias   = mod.bias

        self.features = self.a.shape[0]
        assert self.a.shape == (features,)

        self.log_inv_std_weight = nn.Parameter(torch.ones(features))

        if self.bias is not None:
            assert self.bias.shape == (features,)
            self.log_inv_std_bias = nn.Parameter(torch.ones(features))

        self.s = nn.Parameter(torch.ones(self.features, 1))

    def pred_grad_weight(self, gamma_in):
        if gamma_in.G is not None:
            return self.s*gamma_in.G.diagonal()
        else:
            None

    def pred_grad_bias(self, gamma_in):
        return mul_none(self.s, gamma_in.g)

    def check(self, gamma_in, module_inputs):
        if gamma_in.G is not None:
            gamma_in.G.shape == (self.features, self.features)

        if gamma_in.g is not None:
            gamma_in.g.shape == (self.features,)
        
        module_inputs['weight'].shape == (self.features,)
        if 'bias' in module_inputs:
            module_inputs['bias'].shape == (self.features,)

    def gamma_out(self, gamma_in, grads):
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


    def inv_chol_vec(self, gamma_in, grad_module_inputs):
        """
        Converts gradients sampled from the gradient model into IID noise.
        """
        self.check(gamma_in, grad_module_inputs)

        noise_module_outputs = {}

        grad_weight       = grad_module_inputs['weight']
        corr_noise_weight = grad - self.pred_weight_grad(self, gamma_in)
        iid_noise_weight  = self.log_inv_std_weight.exp() * corr_noise_weight 
        noise_module_outputs['weight'] = iid_noise_weight 

        if self.bias is not None:
            grad_bias       = grad_module_inputs['bias']
            corr_noise_bias = grad - self.pred_bias_grad(self, gamma_in)
            iid_noise_bias  = self.log_inv_std_bias.exp() * corr_noise_bias 
            noise_module_outputs['bias'] = iid_noise_bias 

        return (self.gamma_out(gamma_in, grad_module_inputs), noise_module_outputs)

    def chol_vec(self, gamma_in, noise_module_inputs):
        """
        Converts IID noise into gradients from the gradient model.
        """
        self.check(gamma_in, noise_module_inputs)

        grad_module_outputs = {}

        iid_noise_weight  = noise_module_inputs['weight']
        corr_noise_weight = (-self.log_inv_std_weight).exp() * iid_noise_weight
        grad_weight       = corr_noise_weight + self.pred_weight_grad(self, gamma_in)
        grad_module_outputs['weight'] = grad

        if self.bias is not None:
            iid_noise_bias  = noise_module_inputs['bias']
            corr_noise_bias = (-self.log_inv_std_bias).exp() * iid_noise_bias
            grad_bias       = corr_noise_bias + self.pred_bias_grad(self, gamma_in)
            grad_module_outputs['bias'] = grad_bias

        return (self.gamma_out(gamma_in, grad_module_outputs), grad_module_outputs)

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

def kldify(mod):
    if isinstance(mod, modify.ModuleGroup):
        return kld_module_group[type(mod)]({k: kldify(v) for (k, v) in mod.mods.items()})
    elif type(mod) in kld_modules:
        return kld_modules[type(mod)](mod)
    else:
        raise Exception(f"KLD doesn't know how to handle {type(mod)}")
        
        
