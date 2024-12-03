import torch
import torch.nn as nn

import modify

default_beta = 0.1
default_eps = 10E-7

#################################################
#### Changed argument order for mat_vec_prod ####
#################################################

##########################################
#### Type for holding Gamma and gamma ####
##########################################

def mean_except_last(x):
    assert 1 <= x.ndim
    if x.ndim == 1:
        return x
    else:
        return x.mean(tuple(range(x.ndim - 1)))

def gamma_true(grad, x):
    assert grad.shape == x.shape
    features = x.shape[-1]
    g = mean_except_last(grad)[:, None]
    
    x  = x.view( -1, features)
    grad = grad.view(-1, features)

    G = (grad.mT @ x) / x.shape[0]

    return Gamma(G, g)

class AbstractGamma(): pass
    def __init__(self, G):
        assert isinstance(G, (type(None), torch.Tensor))
        if G is not None:
            assert 2 == G.ndim
            assert G.shape[0] == G.shape[1]

        self.G = G
    
class Gamma(AbstractGamma):
    def passthrough(self):
        return PassThroughGamma(self.G)

class PassThroughGamma(AbstractGamma): pass
    """
    Represents a Gamma that does not even attempt to approximate
    Gamma for the current activations.  This will usually occur
    because we have passed Gamma straight through e.g. a ReLU
    without trying to e.g. linearise the ReLU.    
    """
    def passthrough(self):
        return self


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
    assert isinstance(xs, (tuple, AbstractGamma))
    if isinstance(xs, tuple):
        for x in xs:
            validate_tuple_or_gamma(x)

#############################################
#### Basic EMA + linear regression stuff ####
#############################################

class EMA(nn.Module):
    """
    Lazily initialized EMA
    """
    def __init__(self, beta):
        self.beta = beta
        self.ema = None

    def update(self, x):
        if self.ema is None:
            self.register_buffer('ema', t.zeros_like(x))

        assert x.shape == self.ema.shape
        assert x.device == self.ema.device
        assert x.dtype == self.ema.dtype
        self.ema.mul_(self.beta).add_(x, 1-self.beta)

    def forward(self):
        return self.ema

class Buffer(nn.Module):
    """
    Lazily initialized Buffer
    """
    def __init__(self):
        self.value = None

    def update(self, x):
        if self.value is None:
            self.register_buffer('value', x)
        else:
            assert x.shape == self.value.shape
            assert x.device == self.value.device
            assert x.dtype == self.value.dtype
            self.value.copy_(x)

    def forward(self):
        return self.value


class UnivariateLinearRegressionNoBias(nn.Module):
    """
    This is univariate linear regression.  So for each regression
    problem, there is one input feature, and one weight.  But, you 
    solve many of these univariate problems in parallel.

    Specifically, we have a different set of problems for each 
    index in the feature dimension, which appears last in the inputs.

    We treat the other dimensions as giving independent samples for 
    those inference problems.
    """
    def __init__(beta=default_beta, eps=default_eps):
        self.eps = eps

        #Buffers representing expectations of x^2, x*y
        self.xx = EMA(beta)
        self.xy = EMA(beta)
        #Learned weight
        self.w = Buffer()

    def update(self, x, y):
        assert x.shape[-1] = y.shape[-1]
        features = x.shape[-1]
        x = x.view(-1, features)
        y = y.view(-1, features)
        assert x.shape[0] == y.shape[0]

        self.xx.update((x*x).mean(0))
        self.xy.update((x*y).mean(0))
        self.w.update(self.xy() / (self.x2() + self.eps))

    def forward(self, x):
        return x * self.w()

class UnivariateLinearRegressionWithBias(nn.Module):
    """
    This is univariate linear regression.  So for each regression
    problem, there is one input feature, and one weight.  But, you 
    solve many of these univariate problems in parallel.

    Specifically, we have a different set of problems for each 
    index in the feature dimension, which appears last in the inputs.

    We treat the other dimensions as giving independent samples for 
    those inference problems.
    """
    def __init__(self, beta=default_beta, eps=default_eps):
        super().__init__()
        self.eps = eps

        #Buffers representing expectations of x, y, x^2, x*y
        self.x  = EMA(beta)
        self.y  = EMA(beta)
        self.xx = EMA(beta)
        self.xy = EMA(beta)
        #Learned weight and bias
        self.w = Buffer()
        self.b = Buffer()

    def update(self, x, y):
        assert x.shape[-1] = y.shape[-1]
        features = x.shape[-1]
        x = x.view(-1, features)
        y = y.view(-1, features)
        assert x.shape[0] == y.shape[0]

        self.x.update(x.mean(0))
        self.y.update(y.mean(0))
        self.xx.update((x*x).mean(0))
        self.xy.update((x*y).mean(0))

        self.w.copy_((self.xy() - self.x()*self.y()) / (self.xx() - self.x()*self.x() + self.eps))
        self.b.copy_(self.y() - self.w()*self.x())

    def forward(self, x):
        assert x.shape[-1] == self.features
        return x * self.w() + self.b()

class LinearRegressionR(nn.Module):
    """
    Multivariate linear regression.
    Y = X @ WR

    In general, 3 dimensional inputs:
    [separate_problems, samples, event_dim]
    """
    def __init__(self, beta=default_beta, eps=default_eps):
        super().__init__()
        self.eps = eps

        self.XTX = EMA(beta)
        self.XTY = EMA(beta)
        self.W = Buffer()

    def update(self, X, Y):
        assert 2 == len(X.shape)
        assert 2 == len(Y.shape)
        assert X.shape[0] == Y.shape[0]

        in_features = X.shape[-1]
        out_features = Y.shape[-1]

        self.XTX.update(X.mT @ X)
        self.XTY.update(X.mT @ Y)

        XTX_plus_eps = self.XTX() + self.eps * t.eye(in_features, dtype=X.dtype, device=X.device)
        chol_XTX_plus_eps = t.cholesky(XTX_plus_eps)
        
        # Solves XTX^{-1} XTY
        self.W.update(torch.cholesky_solve(self.XTY(), chol_XTX_plus_eps)

    def forward(self, X):
        return x @ self.W()

class LinearRegressionL(nn.Module):
    """
    Multivariate linear regression.
    Y = WL @ X

    In general, 3 dimensional inputs:
    [separate_problems, event_dim, samples]
    """
    def __init__(self, beta=default_beta, eps=default_eps):
        super().__init__()
        self.linear_regression_r = LinearRegressionR(beta=beta, eps=eps)

    def forward(self, X):
        return self.linear_regression_r(X.mT).mT

    def update(self, X, Y):
        self.linear_regression_r.update(X.mT, Y.mT)

class KFacLinearRegression(nn.Module):
    """
    Solves 
    Y = WL @ X @ WR
    Basic strategy: coordinate descent on two separate linear regression problems.
    Optimizing WR with,
    Y = (WL @ X) @ WR
    treating WL@X as the inputs.  And optimize WL with,
    Y = WL @ (X @ WR)
    treating X@WR as the inputs.

    In general, 4 dimensional inputs:
    [separate_problems, samples, event_dim1, event_dim2]
    """
    def __init__(self, beta=default_beta, eps=default_eps):
        super().__init__()

        self.lin_reg_l = LinearRegressionL(beta=beta, eps=eps)
        self.lin_reg_r = LinearRegressionR(beta=beta, eps=eps)
        self.initialized = False

    def update(self, X, Y):
        #X and Y can be totally different shapes.
        assert 2 == len(X.shape)
        assert 2 == len(Y.shape)

        #Problem: given lazy initialization, can only run forward on LinearRegressionL/R
        #After its been updated.  But we need to run forward on LinearRegressionL to 
        #update LinearRegressionR.  Solution is to fake the first forward using randomly
        #initialized weights.
        if not self.initialized:
            WL = t.randn(Y.shape[0], X.shape[0], device=X.device, dtype=X.device) * t.rsqrt(X.shape[0])
            WR = t.randn(X.shape[1], Y.shape[1], device=X.device, dtype=X.device) * t.rsqrt(X.shape[1])
            self.lin_reg_l.update(     X @ WR, Y)
            self.lin_reg_r.update(WL @ X,      Y)
            self.initialized = True
            
        self.lin_reg_l.update(self.lin_reg_r(X), Y)
        self.lin_reg_r.update(self.lin_reg_l(X), Y)

    def forward(self, X):
        return self.lin_reg_l(self.lin_reg_r(X))

def tensor_sum(xs):
    total = xs[0]
    for x in xs[1:]:
        total = total + x
    return total

class AbstractMultiInputLR(nn.Module):
    """
    Solves 
    Y = f(X_1) + f(X_2)
    f is anything with a f.forward(...) and f.update(...) method.  e.g.:
        LinearRegressionR
        LinearRegressionL
        KFacLinearRegression

    For the update, we do 
    """
    def __init__(self, beta=default_beta, eps=default_eps):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.initialized=False

    def preprocess_inputs(Xs):
        """
        We usually expect the inputs to be a tuple of tensors. But they may also be a single tensor.  
        In that case, we wrap the single tensor in a tuple.
        """
        assert isinstance(Xs, (tuple, torch.Tensor))
        if isinstance(Xs, torch.Tensor):
            Xs = (Xs,)
        return Xs

    def update(self, Xs, Y):
        Xs = preprocess_inputs(Xs)

        assert 2 == len(Y.shape)
        assert 1 <= len(Xs)
        assert all(2 == len(X.shape) for X in Xs)
        assert all(Y.shape[0] == X.shape[0] for X in Xs)

        #Problem: given lazy initialization, can only run forward on LinearRegressionR
        #after its been updated.  But we need to run forward on the other LinearRegressionR's
        #first.  Solution is to assume other weights are zero for the first forward pass.
        if not self.initialized:
            self.lin_regs = [self.LR(beta=beta, eps=eps) for _ in Xs]
            for i in range(Xs):
                #Divide by len(Xs) so we get the right answer on the first step for simple problems
                self.lin_regs[i].update(Xs[i], Y/len(Xs)) 
            self.initialized=True

        preds = self.preds(Xs)

        for i in range(Xs):
            other_preds = [pred for (j, pred) in enumerate(preds) if i != j]
            self.lin_regs[i].update(Xs[i], Y - tensor_sum(other_preds))
            #After updating lin_regs[i], update the corresponding prediction.
            #May not be necessary, but will give faster + more stable convergence.
            preds[i] = self.lin_regs(Xs[i])

    def preds(self, Xs):
        assert len(Xs) == len(self.lin_regs)
        return [self.lin_regs[i](Xs[i]) for i in range(len(self.kfac_lin_regs))]

    def forward(self, Xs):
        Xs = self.preprocess_inputs(Xs)
        return tensor_sum(self.preds(Xs))

class MultiInputKFacLinearRegression(AbstractMultiInputLR):
    LR = KFacLinearRegression

class MultiInputLinearRegressionR(AbstractMultiInputLR):
    LR = LinearRegressionR

class MultiInputLinearRegressionL(AbstractMultiInputLR):
    LR = LinearRegressionL



####################
#### Superclass ####
####################

class KLD():
    """
    Inherited by all modules that do any wrapping.

    All wrapped modules have:
    forward (as usual, takes standard inputs).
    * Not strictly necessary, as we could use the forward method from the original unwrapped network
    * However, it is useful, as it allows us to insert hooks to e.g. compute the true Gammas.

    Three "types" of method:
    * Non-underscore methods.  These are user-accessible.  They just take a vector (represented as a dict)
      with the smae shape as the parameters, and return a vector of the same shape.  They insert gamma_in=None.
    * Single-underscore methods.  These check the input, and pass control on to the double underscore methods.

    chol_vec and inv_chol_vec:
    * computes the product of a vector with the Cholesky or inverse Cholesky of the Fisher (i.e. the covariance of the gradients)

    This superclass just defines the "wrappers": chol_vec, inv_chol_vec, update.

    Still need to implement:
    * forward (just like standard forward).
    * _chol_vec     (gamma_in, vec_in -> gamma_out, vec_out)
    * _inv_chol_vec (gamma_in, vec_in -> gamma_out, vec_out)
    * _update       (gamma_in, vec_in -> gamma_out)
    * check_inputs  (gamma_in, vec_in -> None

    Note that that you don't need to have vec_in (i.e. gradients) as an input to _update.  You could
    just use gradients on the parameters.  The reason for doing it this way is because you probably
    want to use a common gamma_out (gamma_in, vec_in -> gamma_out) to take vec_in as an input. 
    
    """
    def __init__(self, mod):
        super().__init__()
        self.mod = mod
        self.init()

    def init(self):
        pass

    def forward(self, *args, **kwargs):
        self.mod(*args, **kwargs)

    def chol_vec(self, vec_in):
        """
        vec_in -> vec_out
        """
        gamma_out, vec_out = self._chol_vec(Gamma(None), vec_in)
        return vec_out

    def inv_chol_vec(self, vec_in):
        """
        vec_in -> vec_out
        """
        gamma_out, vec_out = self._inv_chol_vec(Gamma(None), vec_in)
        return vec_out

    def update(self, vec_in):
        """
        vec_in (representing gradients) -> None
        """
        self._update(Gamma(None))



#################################
#### Parallel and Sequential ####
#################################

class ModuleGroup(KLD):
    """
    Only used for Parallel and Sequential.

    Doesn't define forward (its going to come by inheritence in Parallel and Sequential).
    """
    def _chol_vec(self, gamma_in, vec_in):
        return self.mat_vec_prod('_chol_vec', gamma_in, vec_in)
    def __inv_chol_vec(self, gamma_in, vec_in):
        return self.mat_vec_prod('_inv_chol_vec', gamma_in, vec_in)

class Parallel(ModuleGroup, modify.Parallel):
    def check_inputs(gamma_ins, vec_in):
        assert isinstance(vec_in, dict)
        validate_tuple(gamma_ins)

    def __update(self, gamma_ins, vec_in):
        self.check_inputs(gamma_ins, vec_in)

        gamma_outs = []
        for ((name, mod), gamma_in) in zip(self.mods.items(), gamma_ins):
            gamma_outs.append(mod._update(gamma_in, vec_in[name]))
        return gamma_outs

    def mat_vec_prod(self, method, gamma_ins, vec_in):
        self.check_inputs(gamma_ins, vec_in)

        vec_outs = {}
        gamma_outs = []
        for ((name, mod), gamma_in) in zip(self.mods.items(), gamma_ins):
            gamma_out, vec_out = getattr(mod, method)(gamma_in, vec_in[name])
            vec_outs[name] = vec_out
            gamma_outs.append(gamma_out)
        return (gamma_outs, vec_outs)
        
class Sequential(ModuleGroup, modify.Sequential):
    def check_inputs(gamma, vec_in):
        assert isinstance(vec_in, dict)
        validate_tuple_or_gamma(gamma)

    def __update(self, gamma, vec_in):
        self.check_inputs(gamma, vec_in):

        for name, mod in self.mods.items():
            gamma = mod._update(gamma, vec_in[name])
        return gamma

    def mat_vec_prod(self, method, gamma, vec_in):
        self.check_inputs(gamma, vec_in):

        vec_outs = {}
        for name, mod in self.mods.items():
            gamma, vec_out = getattr(mod, method)(gamma, vec_in[name])
            vec_outs[name] = vec_out
        return (gamma, vec_outs)



##########################
#### No Param Modules ####
##########################

class LeafModule(KLD):
    """
    For stuff like nonlinearities, Copy, Add, RMSNorm, LayerNorm.
    All these modules don't have any parameters, so don't have any
    gradients, and hence no module_inputs or module_results.  
    All we need to do is propagate the Gamma.

    This is an abstract class; concrete classes must override:
    * __inv_chol_vec (gamma_in, vec_in -> vec_out)
    * __chol_vec     (gamma_in, vec_in -> vec_out)
    * __update       (gamma_in, vec_in -> None)
    * gamma_out      (gamma_in, vec_in -> gamma_out)

    They may override: 
    * init
    * forward
    """
    def _update(self, gamma_in, vec_in):
        self.__update(self, gamma_in, vec_in)
        return self.gamma_out(gamma_in, vec_in)

    def _inv_chol_vec(self, gamma_in, vec_in):
        return (self.gamma_out(gamma_in, vec_in), self.__inv_chol_vec(gamma_in, vec_in))

    def _chol_vec(self, gamma_in, vec_in):
        return (self.gamma_out(gamma_in, vec_in), self.__inv_chol_vec(gamma_in, vec_in)

class NoParamModule(LeafModule):
    """
    For stuff like nonlinearities, Copy, Add, RMSNorm, LayerNorm.
    All these modules don't have any parameters, so don't have any
    gradients, and hence no module_inputs or module_results.  
    All we need to do is propagate the Gamma.

    This is an abstract class; concrete classes must override:
    * __update (gamma_in, vec_in -> None)
    * gamma_out (gamma_in, vec_in -> gamma_out)

    They may override: 
    * init
    * forward
    """
        
    def check_inputs(self, gamma, vec):
        assert isinstance(gamma, AbstractGamma)
        assert isinstance(vec, dict) and (0==len(vec))

    def __inv_chol_vec(self, gamma_in, vec_in):
        return {}

    def __chol_vec(self, gamma_in, vec_in):
        return {}

class PassthroughModule(NoParamModule):
    """
    Takes mod (a module with a single input and single output tensor, such
    as ReLU), and wraps it.  It doens't modify Gamma, so doesn't do any updates:
    it just passes through the gamma.
    """
    def __update(self, gamma_in, vec_in):
        pass

    def gamma_out(self, gamma_in, vec_in):
        return gamma_in.passthrough()

class KFacModule(NoParamModule):
    """
    Can wrap any module that returns a single tensor.  It may take as input a single
    or multiple tensors.

    It does a full KFAC linear regression from input Gamma to output Gamma.

    Note that we use the actual Gamma_out, and the estimate/input Gamma_in.  That's
    because we don't have access to the actual Gamma_in when we're actually running
    we only have access to the estimated/input Gamma_in.

    These modules are likely to be inefficient, but useful for testing.
    """

    def init(self):
        self.y = None
        self.gamma_out = None

        self.lin_reg = MultiInputKFacLinearRegression()

        def hook(mod, grad_output):
            #Compute Gammas at the input and output.
            self.gamma_out = self.y.mT @ grad_output
            #Delete stored x and y.
            self.y = None

        self.register_full_backward_pre_hook(hook)

    def forward(self, *args, **kwargs):
        y = self.mod(*args, **kwargs)
        self.y = y.detach()
        return y

    def __update(self, gamma_in, vec_in):
        self.lin_reg.update(gamma_in, self.gamma_out)
        self.gamma_out = None

    def gamma_out(self, gamma_in, vec_in):
        return Gamma(self.kfac_lin_reg(gamma_in))
        

class KFacCopy(NoParamModule):
    """
    Copy is the only single input, multiple output module.  So we may as well handle this separately.
    """
    def init(self):
        self.y = None
        self.gamma_outs = None

        self.lin_regs = [KFacLinearRegression() for _ in mod.copies]

        def hook(mod, grad_outputs):
            #Compute Gammas at the input and output.
            self.gamma_outs = [y.mT @ go for (y, go) in zip(self.ys, grad_outputs)]
            #Delete stored y.
            self.y = None

        self.register_full_backward_pre_hook(hook)

    def gamma_out(self, gamma_in, vec_in):
        return [Gamma(lin_reg(gamma_in)) for lin_reg in self.lin_regs]

    def forward(self, x):
        self.y = x.detach() # As this is a copy module, x == ys[i]
        return self.mod(*args, **kwargs)

    def __update(self, gamma_in, vec_in):
        for i in range(self.mod.copies):
            self.lin_reg[i].update(gamma_in, self.gamma_outs[i])
        self.gamma_outs = None



#################################
#### Classes with parameters ####
#################################

class Linear(KLD):
    """
    There are two prediction problems here:
       gamma_in -> grad_w
       (gamma_in, grad_w) -> gamma_out
    (Note that this second linear regression would usually only depend on 
    """
    def __init__(self, mod, indep_across_layers: bool):
        super().__init__()
        assert isinstance(mod, nn.Linear)
        self.mod = mod
        self.indep_across_layers = indep_across_layers

        self.out_features, self.in_features = self.mod.weight.shape
        self.in_features_bias = self.in_features + int(mod.bias is not None)
        if self.bias is not None:
            assert self.bias.shape == (self.out_features,)

        self.lin_reg_grad_w = KFacLinearRegression()
        self.lin_reg_grad_b = KFacLinearRegression()
        self.lin_reg_gamma_out = MultiInputKFacLinearRegression()

        def hook(mod, grad_output):
            #Compute Gammas at the input and output.
            self.gamma_out = self.y.mT @ grad_output
            #Delete stored x and y.
            self.y = None

        self.register_full_backward_pre_hook(hook)

        #Inverses of the Kronecker factored Cholesky of the covariance matrix.
        #chol^{-1}(cov_left), i.e. lower-triangular matrix, representing the 
        #inverse of the Cholesky of the left Kronecker factor of the covariance.
        self.register_buffer('invL', t.eye(self.out_features))
        #chol^{-T}(cov_right) i.e. upper-triangular matrix, representing the
        #transpose of the inverse of the Cholesky of the right Kronecker factor of the covariance.
        self.register_buffer('invU', t.eye(self.in_features_bias))

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

    def gamma_out(self, gamma_in, vec_in):
        G = grads['weight'] @ self.weight.mT
        if 'bias' in grads:
            g = grads['bias'][:, None]
        elif gamma_in.g is not None:
            g = self.inv_weight_T @ gamma_in.g
        else:
            g = None
        return Gamma(G, g)

    def pred_grad_w(self, gamma_in, grad_w):
        #gamma_in.G is  in_features x in_features or None
        #grad_w     is out_features x in_features
        if self.indep_across_layers or (gamma_in.G is None)
            return torch.zeros_like(grad_bias)
        else:
            return self.lin_reg_grad_w(gamma_in.G)

    def pred_grad_b(self, gamma_in, grad_b):
        #gamma_in.G is  in_features x in_features or None
        #grad_b     is out_features x 1
        if self.indep_across_layers or (gamma_in.G is None)
            return torch.zeros_like(grad_b)
        else:
            return self.lin_reg_grad_b(gamma_in.G)

    def __update(self, gamma_in):
        self.lin_reg.update(gamma_in, self.gamma_out)
        self.gamma_out = None


    def _inv_chol_vec(self, grad_module_inputs, gamma_in):
        """
        Converts gradients sampled from the gradient model into IID noise.
        """
        self.check(grad_module_inputs, gamma_in)

        grad      = grad_module_inputs['weight']     # out_features x in_features
        pred_grad = self.pred_grad_w(gamma_in, grad) # out_features x in_features

        if self.bias is not None:
            grad_b = grad_module_inputs['bias'][:, None]        # out_features x 1
            pred_grad_b = self.pred_bias_grad(gamma_in, grad_b) # out_features x 1
            grad      = torch.cat((     grad,      grad_b), -1) # out_features x in_features+1
            pred_grad = torch.cat((pred_grad, pred_grad_b), -1) # out_features x in_features+1

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

        iid_noise = noise_module_inputs['weight']              # out_features x in_features
        pred_grad = self.pred_weight_grad(gamma_in, iid_noise) # out_features x in_features

        if self.bias is not None:
            iid_noise_b = noise_module_inputs['bias'][:, None]       # out_features x 1
            pred_grad_b = self.pred_bias_grad(gamma_in, iid_noise_b) # out_features x 1 or None
            iid_noise = torch.cat((iid_noise, iid_noise_b), -1)      # out_features x in_features+1
            pred_grad = torch.cat((pred_grad, pred_grad_b), -1)      # out_features x in_features+1

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
