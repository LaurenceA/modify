#TODOs:
#  
#Testing:
#  Test all the linear regression stuff (generate data with the right distribution).
#  Test inv_chol, chol on CovLinearRegression
#  

import math
import torch
import torch.nn as nn
from torch.linalg import triangular_solve

import modify

default_beta = 0.1
default_eps = 1E-7

#################################################
#### Changed argument order for mat_vec_prod ####
#################################################

##########################################
#### Type for holding Gamma and gamma ####
##########################################

def true_gamma(output, grad, has_bias):
    """
    Returns a Tensor, not an AbstractGamma
    """
    assert isinstance(output, torch.Tensor)
    assert isinstance(grad, torch.Tensor)
    assert isinstance(has_bias, bool)

    self.output = self.output.view(-1, self.output.shape[-1])
    self.grad = self.grad.view(-1, self.grad_output.shape[-1])
    gamma_out = self.output.mT @ grad_output / math.sqrt(self.output.shape[0]) #out_features   x out_features
    if has_bias:
        gamma_out = t.cat((gamma_out, grad.mean(0)), 0)                        #out_features +1   x out_features
    return gamma_out                                                           #out_features(+1?) x out_features

class AbstractGamma(): pass

class SomeGamma(AbstractGamma):
    """
    This contains information about the previous gradients that we pass forward
    to later modules.

    Specifically, Gamma represents a prediction of:
    Gamma = X^T dL/dX

    Thus, without a bias Gamma is a square, features x features matrix.

    With a bias, it is a features+1 x features matrix, as we include the bias
    as an extra feature.
    """
    def __init__(self, G):
        self.G = G
        assert 2 == self.G.ndim
        self.features = self.G.shape[1]
        assert self.G.shape[0] in (self.features+1, self.features)
        self.has_bias = self.G.shape[0] == self.features+1

class NoneGamma(AbstractGamma): 
    """
    Passed in to the network initally.  Implies that there is no information from
    previous module's gradients to pass forward (because we're so early in the
    network that there haven't been any previous modules yet).
    """
    pass

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
    Lazily initialized exponential moving average (EMA).
    """
    def __init__(self, beta):
        self.beta = beta
        self.initialized = False

    def update(self, x):
        if not self.initialized:
            self.register_buffer('ema', t.zeros_like(x))
            self.initialized = True

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
        self.initialized = False

    def update(self, x):
        if not self.initialized:
            self.register_buffer('value', x)
            self.initialized = True

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
        self.w.update(self.xy() / (self.xx() + self.eps))

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
    """
    def __init__(self, beta=default_beta, eps=default_eps):
        super().__init__()
        self.lin_reg_r = LinearRegressionR(beta=beta, eps=eps)

    def forward(self, X):
        return self.lin_reg_r(X.mT).mT

    def update(self, X, Y):
        self.lin_reg_r.update(X.mT, Y.mT)

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
        #
        #Note that we only need to do this fake initialization for one side.  As once we
        #have initialized one side, we can use it to initialize the other side.
        if not self.initialized:
            WL = t.randn(Y.shape[0], X.shape[0], device=X.device, dtype=X.device) / math.sqrt(X.shape[0])
            self.lin_reg_r.update(WL @ X,      Y)
            self.initialized = True
            
        self.lin_reg_l.update(self.lin_reg_r(X), Y)
        self.lin_reg_r.update(self.lin_reg_l(X), Y)

    def forward(self, X):
        return self.lin_reg_l(self.lin_reg_r(X))

class NoneRegression(nn.Module):
    """
    Has the same interface as the other regression methods, but takes None, and returns zeros.
    """
    def __init__(self):
        self.zeros = Buffer()

    def update(X,Y):
        assert X is None
        self.zeros.update(t.zeros_like(Y))

    def forward(X):
        assert X is None
        return self.zeros()


class CovLinearRegression(nn.Module):
    """
    Takes a linear regression module, and gives a Kronecker factored estimate of
    the covariance of the outputs.

    X ~ MN(0, U, V)
    E[X X'] = U Tr(V)
    E[X' X] = V Tr(U)
    T = Tr(E[X X']) = Tr(E[X' X]) = X.square().sum() = Tr(U) Tr(V)
    U = E[X X'] / sqrt(T)
    V = E[X' X] / sqrt(T)
    """
    def __init__(self, lin_reg, beta=default_beta):
        super().__init__()
        self.lin_reg = lin_reg
        self.EET = EMA(beta=beta)
        self.ETE = EMA(beta=beta)
        self.T   = EMA(beta=beta)

        self._chol_left = None
        self._chol_right = None

    def update(X, Y):
        assert 2 == X.ndim
        assert 2 == Y.ndim
        self.lin_reg.update(X, Y)
        E = Y - self.lin_reg(X)
        self.EET.update(E@E.mT)
        self.ETE.update(E.mT@E)
        self.T.update(E.square().sum())

        #Wipe cached cholesky
        self.chol_left = None
        self.chol_right = None

    def compute_chol(self):
        T_rsqrt = self.T().rsqrt()
        if self.chol_left is None:
            self.chol_left  = t.cholesky(self.EET() * T_rsqrt)
        if self.chol_right is None:
            self.chol_right = t.cholesky(self.EET() * T_rsqrt)

    def chol_vec_noise(self, iid_niose):
        """
        Converts IID noise into correlated noise.
        """
        assert 2 == iid_noise.ndim
        self.compute_chol()
        corr_noise = chol_left @ iid_noise @ chol_right.mT
        return corr_noise

    def inv_chol_vec_noise(self, corr_noise)
        """
        Converts correlated noise to IID noise.
        Inverse of chol_vec_noise.
        """
        assert 2 == corr_noise.ndim
        self.compute_chol()
        # chol_left_iid_noise = chol_left @ iid_noise = corr_noise @ chol_right^{-T} = (chol_right^{-1} @ corr_noise^T)^T
        chol_left_iid_noise = triangular_solve(chol_right, corr_noise.mT, upper=False).mT
        iid_noise = triangular_solve(chol_left, chol_left_iid_noise, upper=False)
        return iid_noise

    def chol_vec(self, X, iid_noise)
        """
        Converts iid noise to a sample of the gradient
        """
        corr_noise = self.chol_prod_noise(iid_noise)
        grad = self.lin_reg(X) + corr_noise
        return grad

    def inv_chol_vec(self, X, grad)
        """
        Converts a sample of the gradient to IID noise.
        Inverse of chol_vec.
        """
        corr_noise = grad - self.lin_reg(X)
        iid_noise = self.inv_chol_prod_noise(corr_noise)
        return iid_noise

    def forward(X):
        assert 2 == X.ndim
        return self.lin_reg(X)
         

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

    The current implementation is quite naive, as we do coordinate descent on each function in sequence.

    In the long-run, we might want to do this as one big KFAC, with structured inputs:
                     X_1   0    WR_1
    Y = (WL_1 WL_2) (        ) (    ) = WL_1 X_1 WR_1^T + WL_2 X_2 WR_2^T
                      0   X_2   WR_2
    To do this operation, we first compute,
     X_1   0    WR_1     X_1 WR_1
    (        ) (    ) = (        )
      0   X_2   WR_2     X_2 WR_2
    Then,
                     X_1 WR_1
    Y = (WL_1 WL_2) (        )
                     X_2 WR_2 
    The inputs to the first problem are a block diagonal matrix, while the inputs to the second problem are
    a "block-vector".  Now, the problem is that we actually do learning using the second problem and the second
    problem doesn't have any exploitable structure in the inputs, at least if you don't assume X_1 is independent
    of X_2.

    In some ways though, this is fine:  
    * We would usually this to combine Parallel branches and we should be considering correlations across the 
      branches.
    * We only ever (in transformers) have two branches, so the increase in costs should be okay.

    There is also a problem for the interface.  But that is easily resolvable.  Specifically, just provide two 
    apply methods:
    apply_block_diag
    apply_block_vec
    """
    def __init__(self, beta=default_beta, eps=default_eps):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.initialized = False

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
            Y_over_lenXs = Y/len(Xs)
            for i in range(Xs):
                #Divide by len(Xs) so we get the right answer on the first step for simple problems
                self.lin_regs[i].update(Xs[i], Y_over_lenXs)
            self.initialized=True

        preds = self.preds(Xs)

        for i in range(Xs):
            other_preds = [*preds[:i], preds[:i+1]]
            self.lin_regs[i].update(Xs[i], Y - tensor_sum(other_preds))
            #After updating lin_regs[i], update the corresponding prediction.
            #May not be necessary, but will give faster + more stable convergence.
            preds[i] = self.lin_regs(Xs[i])

    def preds(self, Xs):
        return [lin_regs(X) for (lin_reg, X) in zip(self.lin_regs, Xs)]

    def forward(self, Xs):
        Xs = self.preprocess_inputs(Xs)
        preds = self.preds(Xs)
        return tensor_sum(preds)

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
      with the same shape as the parameters, and return a vector of the same shape.  They pass gamma_in=NoneGamma() downstream
    * Single-underscore methods.  These check the input, and pass control on to the double underscore methods
      implemented by concrete classes.

    chol_vec and inv_chol_vec:
    * computes the product of a vector with the Cholesky or inverse Cholesky of the Fisher (i.e. the covariance of the gradients)

    This superclass just defines the "wrappers": chol_vec, inv_chol_vec, update.

    Still need to implement:
    * forward (just like standard forward).
    * _chol_vec     (gamma_in, vec_in -> gamma_out, vec_out)
    * _inv_chol_vec (gamma_in, vec_in -> gamma_out, vec_out)
    * _update       (gamma_in, vec_in -> gamma_out)
    * check_inputs  (gamma_in, vec_in -> None

    Note that that we chose to use vec_in in _update to represent the gradients.  You don't need to
    make this choice: you could just use gradients on the parameters.  The reason for doing it this 
    way is because you probably want to use a common gamma_out (gamma_in, vec_in -> gamma_out) which 
    take vec_in as an input. 
    """
    def forward(self, *args, **kwargs):
        return self.mod(*args, **kwargs)

    def chol_vec(self, vec_in):
        """
        vec_in -> vec_out
        """
        gamma_out, vec_out = self._chol_vec(gamma_in=NoneGamma(), vec_in)
        return vec_out

    def inv_chol_vec(self, vec_in):
        """
        vec_in -> vec_out
        """
        gamma_out, vec_out = self._inv_chol_vec(gamma_in=NoneGamma(), vec_in)
        return vec_out

    def update(self, vec_in):
        """
        vec_in (representing gradients) -> None
        """
        gamma_out = self._update(gamma_in=NoneGamma(), vec_in)

#################################
#### Parallel and Sequential ####
#################################

class ModuleGroup(KLD):
    """
    Only used for Parallel and Sequential.

    Uniquely, takes a dict of modules, _not_ the Sequential/Parallel module itself.
    That's because it takes a dict of _wrapped_ modules, not the underlying modules themselves.

    Forward comes from multiple inheritence of modify.Sequential / modify.Parallel.

    Note that this class doesn't register the modules correctly, so e.g. mod.Parameters() won't work.
    """
    def __init__(self, mods, indep_across_layers, initial_layer):
        super().__init__()
        assert isinstance(mods, dict)
        self.mods = mods

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
    For anything that isn't Sequential or Parallel, like Linear, 
    nonlinearities, Copy, Add, RMSNorm, LayerNorm.

    This is an abstract class; concrete classes must override:
    * __init__
    * __inv_chol_vec (gamma_in, vec_in -> vec_out)
    * __chol_vec     (gamma_in, vec_in -> vec_out)
    * __update       (gamma_in, vec_in -> None)
    * gamma_out      (gamma_in, vec_in -> gamma_out)
    * check_inputs   (gamma_in, vec_in -> None)

    They may override: 
    * forward
    """
    def _update(self, gamma_in, vec_in):
        self.check_inputs(gamma_in, vec_in)
        self.__update(self, gamma_in, vec_in)
        return self.gamma_out(gamma_in, vec_in)

    def _inv_chol_vec(self, gamma_in, vec_in):
        self.check_inputs(gamma_in, vec_in)
        return (self.gamma_out(gamma_in, vec_in), self.__inv_chol_vec(gamma_in, vec_in))

    def _chol_vec(self, gamma_in, vec_in):
        self.check_inputs(gamma_in, vec_in)
        return (self.gamma_out(gamma_in, vec_in), self.__inv_chol_vec(gamma_in, vec_in)

class NoParamModule(LeafModule):
    """
    For stuff like nonlinearities, Copy, Add, RMSNorm, LayerNorm,
    but _not_ Linear or ElementwiseAffine.

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
    def __init__(self, mod, indep_across_layers, initial_layer):
        super().__init__()
        assert not initial_layer
        self.mod = mod
        self.init()

    def init(self): 
        pass
        
    def check_inputs(self, gamma_in, vec_in):
        assert isinstance(gamma_in, SomeGamma)
        assert isinstance(vec_in, dict) and (0==len(vec_in))

    def __inv_chol_vec(self, gamma_in, vec_in):
        return {}

    def __chol_vec(self, gamma_in, vec_in):
        return {}

#class PassthroughModule(NoParamModule):
#    """
#    Takes mod (a module with a single input and single output tensor, such
#    as ReLU), and wraps it.  It doens't modify Gamma, so doesn't do any updates:
#    it just passes through the gamma.
#    """
#    def __update(self, gamma_in, vec_in):
#        pass
#
#    def gamma_out(self, gamma_in, vec_in):
#        return gamma_in.passthrough()

class KFacModule(NoParamModule):
    """
    Can wrap any module that returns a single tensor.  It may take as input a single
    or multiple tensors.

    It does a full KFAC linear regression from input Gamma to output Gamma.

    Note that we use the actual gamma_out, and the predicted gamma_in, because the
    predicted gamma_in is what we actually have available at runtime!

    These modules are likely to be inefficient, but useful for testing.
    """

    def init(self):
        self.output = None
        self.grad_output = None

        self.lin_reg = MultiInputKFacLinearRegression()

        def hook(mod, grad_output):
            self.grad_output = grad_output
        self.register_full_backward_pre_hook(hook)

    def forward(self, *args, **kwargs):
        output = self.mod(*args, **kwargs)
        self.output = output.detach()
        return output

    def __update(self, gamma_in, vec_in):
        gamma_out = true_gamma(self.output, self.grad_output, gamma_in.has_bias)
        self.lin_reg(gamma_in, gamma_out)
        #Delete stored output and grad.
        self.output = None
        self.grad_output = None

    def gamma_out(self, gamma_in, vec_in):
        return Gamma(self.kfac_lin_reg(gamma_in))

class KFacCopy(NoParamModule):
    """
    Copy is the only single input, multiple output module.  So we may as well handle this separately.
    """
    def init(self):
        self.output = None #Just one output, as they're all copies.
        self.grad_outputs = None

        #Eventual TODO: do the linear regressions in parallel, rather than separately.
        self.lin_regs = [KFacLinearRegression() for _ in mod.copies]

        def hook(mod, grad_outputs):
            self.grad_outputs = grad_outputs 
        self.register_full_backward_pre_hook(hook)

    def forward(self, x):
        self.output = x.detach() # As this is a copy module, each output is just a copy of the input.
        return self.mod(*args, **kwargs)

    def gamma_out(self, gamma_in, vec_in):
        return [Gamma(lin_reg(gamma_in)) for lin_reg in self.lin_regs]

    def __update(self, gamma_in, vec_in):
        for grad_output, lin_reg in zip(self.grad_outputs, self.lin_regs):
            gamma_out = true_gamma(self.output, grad_output, gamma_in.has_bias)
            lin_reg.update(gamma_in, gamma_outs)
        self.output = None
        self.grad_outputs = None



#################################
#### Classes with parameters ####
#################################

class Linear(LeafModule):
    """
    There are two prediction problems here:
       gamma_in -> grad_w
       (gamma_in, grad_w) -> gamma_out
    (Note that this second linear regression would usually only depend on 
    """
    def __init__(self, mod, indep_across_layers: bool, initial_layer:bool):
        self.mod = mod
        self.indep_across_layers = indep_across_layers
        self.initial_layer = initial_layer
        self.zero_pred = initial_layer or indep_across_layers

        self.has_bias = mod.bias is not None
        self.out_features, self.in_features = mod.weight.shape
        self.in_features_bias = self.in_features + int(self.has_bias)
        if self.has_bias:
            assert self.mod.bias.shape == (self.out_features,)

        self.lin_reg_grad = CovLinReg(None if self.zero_pred else KFacLinearRegression())
        self.lin_reg_gamma_out = MultiInputKFacLinearRegression()

        self.output = None
        self.grad_output = None

        def hook(mod, grad_output):
            self.grad_output = grad_output
        self.register_full_backward_pre_hook(hook)

    def forward(self, *args, **kwargs):
        output = self.mod(*args, **kwargs)
        self.output = output.detach()
        return output

    def check_inputs(self, gamma_in, vec_in):
        assert isinstance(vec_in, dict)
        assert isinstance(gamma_in, AbstractGamma)

        if self.initial_layer:
            assert isinstance(gamma_in, NoneGamma)
        else:
            assert isinstance(gamma_in, SomeGamma)
            assert gamma_in.features == self.in_features

        assert 'weight' in vec_in
        assert vec_in['weight'].shape == (self.out_features, self.in_features)
        if 'bias' in vec_in:
            assert vec_in['weight'].shape == (self.out_features,)

    def vec2tensor(self, vec_in):
        """
        Converts the dict vec_in to a tensor, including both the component
        from the weight and the bias.
        """
        assert isinstance(vec, dict)
        tensor = vec['weight'] # out_features x in_features

        assert self.has_bias == ('bias' in vec_in)

        if self.bias:
            tensor_b = vec_in['bias']                  #out_features
            tensor_b = result_b[:, None]               #out_features x 1
            tensor = torch.cat((tensor, tensor_b), -1) #out_features x in_features+1

        return tensor                                  #out_features x in_features(+1?)

    def tensor2vec(self, tensor):
        """
        Converts the dict vec_in to a tensor, including both the component
        from the weight and the bias.
        """
        assert tensor.shape == (self.out_features, self.in_features_bias)

        if self.has_bias:
            return {'weight': tensor[..., :-1], 'bias': tensor[..., -1]}
        else
            return {'weight': tensor}

    def inputs_for_predicting_gamma_out(self, gamma_in, vec_in):
        tensor_vec_in = self.tensor_vec_in(vec_in)
        inputs = [tensor_vec_in]
        if not self.initial_layer:
            inputs.append(gamma_in.G)
        return inputs

    def gamma_out(self, gamma_in, vec_in):
        return self.lin_reg_gamma_out(self.inputs_for_predicting_gamma_out(gamma_in, vec_in))

    def __update(self, gamma_in, vec_in):
        gamma_out = true_gamma(self.output, self.grad_output, gamma_in.has_bias or self.has_bias)
        self.lin_reg_gamma_out.update(self.inputs_for_predicting_gamma_out(gamma_in, vec_in), gamma_out)

        tensor_vec_in = self.tensor_vec_in(vec_in) #Represents gradient of weight and optionally bias.
        self.lin_reg_grad.update(gamma_in.G, tensor_vec_in)

    def __inv_chol_vec(self, gamma_in, vec_in):
        """
        Converts gradients sampled from the gradient model into IID noise.
        """
        tensor_in  = self.vec2tensor(vec_in)
        tensor_out = self.lin_reg_grad.inv_chol_vec(tensor_in)
        vec_out = self.tensor2vec(tensor_out)
        return vec_out

    def __chol_vec(self, noise_module_inputs, gamma_in):
        """
        Converts IID noise into gradients from the gradient model.
        """
        tensor_in  = self.vec2tensor(vec_in)
        tensor_out = self.lin_reg_grad.chol_vec(tensor_in)
        vec_out = self.tensor2vec(tensor_out)
        return vec_out
    
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

kld_modules = {
    nn.Linear: Linear, #Explicit implementation.
    modify.Copy: Copy, #Explicit implementation.
    modify.ElementwiseNonlin: KFacModule, #ElementwiseNonlin,
    modify.Add: KFacModule, #Add,
    modify.Mul: KFacModule, #Mul,
    #nn.RMSNorm: rms_norm,
    #nn.LayerNorm: layer_norm,
}

def kldify(mod, indep_across_layers=False, initial_layer=True):
    if isinstance(mod, modify.Sequential):
        mods = {}
        for (i, (k, v)) in enumerate(mod.mods.items()):
            mods[k] = kldify(v, indep_across_layers, i==0)
        return Sequential(mods)
    elif isinstance(mod, modify.Parallel):
        return Parallel({k: kldify(v, indep_across_layers, initial_layer) for (k, v) in mod.mods.items()})
    elif type(mod) in kld_modules:
        return kld_modules[type(mod)](mod, indep_across_layers, initial_layer)
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
