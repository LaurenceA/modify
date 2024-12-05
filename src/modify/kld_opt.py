
class ElementwiseNonlin(NoParamModule):
    """
    Does linear regression:
        from inputs -> outputs.
        from gradient of inputs -> gradient of outputs.
    Thus, doesn't save any gammas.
    """
    def __init__(self, mod, indep_across_layers, beta=0.9, eps=1E-5):
        super().__init__()
        assert isinstance(mod, modify.ElementwiseNonlin)
        self.mod  = mod
        self.beta = beta
        self.eps  = eps

        #Stuff that isn't initialized immediately, because we don't know how many features
        #there are going to be. Note that this doesn't cause the usual problems with lazily
        #initialized parameters (e.g. params not included in optimizers), as there aren't
        #any parameters here, just buffers.
        self.features      = None
        self.input_lin_reg = None
        self.grad_lin_reg  = None

        def hook(mod, grad_input, grad_output):
            #Update gradient linear regression.
            mod.grad_lin_reg.update(grad_output, grad_intput)
        self.register_full_backward_hook(hook)

    @property
    def features(self):
        return self.mod.features

    def forward(self, x):
        if self.features is None:
            #Finish initialization once number of features is known.
            self.features = x.shape[-1]
            tensor_kwargs = {'device' = x.device, 'dtype' = x.dtype}
            self.input_lin_reg = self.UnivariateLinearRegressionWithBias(self.features, self.beta, self.eps, tensor_kwargs)
            self.grad_lin_reg  = self.UnivariateLinearRegressionNoBias(  self.features, self.beta, self.eps, tensor_kwargs)

        #Update input linear regression.
        y = self.mod(x)
        self.input_lin_reg(x, y)
        return y

    def G_to_G(self, G):
        return  self.grad_lin_reg.w[:, None]  * G * self.input_lin_reg.w

    def g_to_g(self, g):
        return  self.grad_lin_reg.w[:, None]  * g

    def g_to_G(self, g):
        return (self.grad_lin_reg.w[:, None]) * g * self.input_lin_reg.b

    def _update(self, _, gamma_in):
        return self.gamma_out(gamma_in)

class SPDA(NoParamModule):
    def __init__(self, mod, indep_across_layers):
        super().__init__()

    def gamma_out(self, gamma_ins):
        validate_tuple(gamma_ins)
        #Just return gammas from the values.
        return gamma_in[2]

class Copy(NoParamModule):
    """
    Remembering that:
    y_i = x
    dL/dx = \sum_i dL/dy_i
    Gamma^in = <dL/dx x^T>
    Gamma^out = <dL/dy_i y_i^T> = <dL/dy_i x>
    Thus, we regress dL/dx -> dL/dy_i.
    """
    def __init__(self, mod, indep_across_layers):
        super().__init__()
        self.mod = mod
  
        #Stuff that isn't initialized immediately, because we don't know how many features
        #there are going to be.
        self.features = None
        self.grad_lin_regs = None 

        def hook(mod, grad_input, grad_outputs):
            for grad_lin_reg, grad_output in zip(self.grad_lin_regs, grad_outputs):
                grad_lin_reg.update(grad_input, grad_output)
        self.register_full_backward_hook(hook)

    def _init(self, _, gamma_in):
        self.features = gamma_in.features

    def forward(self, x):
        #Finish initialization once number of features is known.
        if self.features is None:
            self.features = x.shape[-1]
            self.grad_lin_regs = [MultivariateLinearRegressionNoBias(self.features, self.features) for _ in range(mod.number_of_copies)]

        return self.mod(x)

    def _gamma_out(self, glr, gamma_in):
        G = None if (gamma_in.G is None) else glr(gamma_in.G.mT).mT 
        g = None if (gamma_in.g is None) else glr(gamma_in.g.mT).mT 
        return Gamma(G, g)

    def gamma_out(self, gamma_in):
        return [self._gamma_out(glr, gamma_in) for glr in grad_lin_regs]

    def _update(self, _, gamma_in):
        return self.gamma_out(gamma_in)

class Add(NoParamModule):
    """
    Problem here is that we're integrating across different input gamma's
    potentially with different noise levels. That means we have to work 
    with the predicted gamma_in propagated through the model, rather than
    the real gamma_in given by the inputs and input gradients.

    Next, the problem is that regressing from multiple input gamma_in tensors
    is super painful.  The simple solution is to learn one coefficient for
    each input gamma.
    """
    def __init__(self, mod, indep_across_layers, beta=0.9, eps=1E-5):
        super().__init__()
        self.G_lin_reg = MultivariateLinearRegressionNoBias(2, 1, copies=None, beta=beta, eps=eps))
        self.g_lin_reg = MultivariateLinearRegressionNoBias(2, 1, copies=None, beta=beta, eps=eps))

        self.y = None
        self.G_out = None
        self.g_out = None

        def hook(mod, grad_y):
            #Saves G_out and g_out
            self.gamma_out = gamma_true(grad_y, self.y)
            self.y = None

        self.register_full_backward_pre_hook(hook)

    def forward(self, xs, gamma_ins):
        assert 2 == len(xs)
        #Save the output
        self.y = self.mod(xs)
        return self.y

    def stacked_Gs_or_gs(self, Ggs):
        """
        Takes a list of gamma_in.G's or gamma_in.g's, and returns None, if all the 
        gamma_in's are None, or returns a single tensor, with zeros for the Nones.
        """
        non_none_Ggs = [Gg for Gg in Ggs if Gg is not None]
        if 0 == len(non_none_Ggs):
            return None
        else:
            Gg0 = non_none_Ggs[0]
            return torch.stack([t.zeros_like(Gg0) if Gg is None else Gg for Gg in Ggs], -1)

    def stacked_Ggs(self, gamma_ins):
        stacked_G_ins = self.stacked_Gs_or_gs([gamma_in.G for gamma_in in gamma_ins]) # None or features x features x 2
        stacked_g_ins = self.stacked_Gs_or_gs([gamma_in.g for gamma_in in gamma_ins]) # None or features x 1        x 2
        return stacked_G_ins, stacked_g_ins

    def _gamma_out(self, stacked_G_ins, stacked_g_ins):
        G_out = None if stacked_G_ins is None else self.G_lin_reg(stacked_G_ins).squeeze(-1)
        g_out = None if stacked_g_ins is None else self.g_lin_reg(stacked_g_ins).squeeze(-1)
        return Gamma(G_out, g_out)

    def gamma_out(self, gamma_ins):
        stacked_G_ins, stacked_g_ins = self.stacked_Ggs(self, gamma_ins)
        return self._gamma_out(stacked_G_ins, stacked_g_ins)

    def _update(self, gamma_ins):
        stacked_G_ins, stacked_g_ins = self.stacked_Ggs(self, gamma_ins)

        if stacked_G_ins is not None:
            self.G_lin_reg.update(stacked_G_ins, self.gamma_out.G)

        if stacked_g_ins is not None:
            self.g_lin_reg.update(stacked_g_ins, self.gamma_out.g)

        self.gamma_out = None

        return self._gamma_out(stacked_G_ins, stacked_g_ins)



class Mul(NoParamModule):
    """
    Similar to Add, we're integrating across different input gamma's
    potentially with different noise levels. That means we have to work 
    with the predicted gamma_in propagated through the model, rather than
    the real gamma_in given by the inputs and input gradients.

    However, the simple solution from Add with just one coefficient for
    each gamma_in won't work, because the multiplicative interactions mean
    that we need different coefficients for each element.

    Our solution is two linear regressions: one for the rows and one for
    the columns.

    To do learning for right_lin_reg regress from the gamma_in's, multiplied by
    coefficients from left_lin_reg to output.

    To do learning for left_lin_reg, do the opposite.
    """
    def __init__(self, mod, indep_across_layers, beta=0.9, eps=1E-5):
        super().__init__()
        self.G_left_lin_reg  = MultivariateLinearRegressionNoBias(2, 1, copies=mod.features, beta=beta, eps=eps)
        self.G_right_lin_reg = MultivariateLinearRegressionNoBias(2, 1, copies=mod.features, beta=beta, eps=eps)
        self.g_lin_reg       = MultivariateLinearRegressionNoBias(2, 1, copies=mod.features, beta=beta, eps=eps)

        self.y = None
        self.G_out = None
        self.g_out = None

        def hook(mod, grad_y):
            #Saves G_out and g_out
            self.gamma_out = gamma_true(grad_y, self.y)
            self.y = None

        self.register_full_backward_pre_hook(hook)

    def G_left_lin_apply(self, Gs):
        return self.G_left_lin_reg.apply(Gs.swap_axes(0,1)).swap_axes(0,1)

    def G_right_lin_apply(self, Gs):
        return self.G_right_lin_reg.apply(Gs)

    def G_left_lin_update(self, Gs):
        return self.G_left_lin_reg.update(Gs.swap_axes(0,1))

    def G_right_lin_update(self, Gs):
        return self.G_right_lin_reg.update(Gs)

    def forward(self, xs, gamma_ins):
        assert 2 == len(xs)
        #Save the output
        self.y = self.mod(xs)
        return self.y

    def stacked_Gs_or_gs(self, Ggs):
        """
        Takes a list of gamma_in.G's or gamma_in.g's, and returns None, if all the 
        gamma_in's are None, or returns a single tensor, with zeros for the Nones.
        """
        non_none_Ggs = [Gg for Gg in Ggs if Gg is not None]
        if 0 == len(non_none_Ggs):
            return None
        else:
            Gg0 = non_none_Ggs[0]
            return torch.stack([t.zeros_like(Gg0) if Gg is None else Gg for Gg in Ggs], -1)

    def stacked_Ggs(self, gamma_ins):
        stacked_G_ins = self.stacked_Gs_or_gs([gamma_in.G for gamma_in in gamma_ins]) # None or features x features x 2
        stacked_g_ins = self.stacked_Gs_or_gs([gamma_in.g for gamma_in in gamma_ins]) # None or features x 1        x 2
        return stacked_G_ins, stacked_g_ins

    def _gamma_out(self, stacked_G_ins, stacked_g_ins):
        G_out = None if stacked_G_ins is None else self.G_lin_reg_apply(stacked_G_ins).squeeze(-1)
        g_out = None if stacked_g_ins is None else self.g_lin_reg_apply(stacked_g_ins).squeeze(-1)
        return Gamma(G_out, g_out)

    def gamma_out(self, gamma_ins):
        stacked_G_ins, stacked_g_ins = self.stacked_Ggs(self, gamma_ins)
        return self._gamma_out(stacked_G_ins, stacked_g_ins)

    def _update(self, gamma_ins):
        stacked_G_ins, stacked_g_ins = self.stacked_Ggs(self, gamma_ins)

        if stacked_G_ins is not None:
            self.G_left_lin_reg_update(self.G_right_lin_reg_apply(stacked_G_ins), self.gamma_out.G)
            self.G_right_lin_reg_update(self.G_left_lin_reg_apply(stacked_G_ins), self.gamma_out.G)

        if stacked_g_ins is not None:
            self.g_lin_reg.update(stacked_g_ins, self.gamma_out.g)

        self.gamma_out = None

        return self._gamma_out(stacked_G_ins, stacked_g_ins)


class MeanSub(NoParamModule):
    """
    Fixed transformation.
    """
    def __init__(self, mod, indep_across_layers):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return self.mod(x)

    def G_to_G(self, G):
        G = G - G.mean(0, keepdim=True)
        G = G - G.mean(1, keepdim=True)
        return G

    def g_to_G(self, g):
        return None

    def g_to_g(self, g):
        return g - g.mean(1, keepdim=True)

class RMSNorm(NoParamModule):
    """
    Needs update: just use mean of inputs...
    """
    def __init__(self, mod, indep_across_layers, eps=1E-5):
        self.mod = mod
        self.eps = eps
        self.xb = EMA(mod.features)

    def forward(self, x):
        self.xb.update(mean_except_last(x))
        return self.mod(x)

    def G_to_G(self, G):
        xb = self.xb
        norm = xb.mT@xb + self.eps
        G = G - xb @ ((xb.mT @ G) / norm)
        G = G - ((G @ xb) / norm) @ xb.mT
        return G

    def g_to_g(self, g):
        xb = self.xb
        norm = xb.mT@xb + self.eps
        return torch.sqrt(norm) * (g - xb @ ((xb.mT @ g) / norm))

    def g_to_G(self, G):
        return None


#################################
#### Classes with parameters ####
#################################

class Linear(nn.Module):
    """
    There are two prediction problems here:
       gamma_in -> grad_w
       (gamma_in, grad_w) -> gamma_out

    The overall strategy is:
        Learn a mapping from grad_x to grad_y.
        That directly tells us how to map:
            gamma_in -> grad_w
            gamma_in -> gamma_out
            grad_w   -> gamma_out

    Then, we do bivariate linear regression to combine: 
        the prediction of gamma_out from gamma_in
        the prediction of gamma_out from grad_w
    """
    def __init__(self, mod, indep_across_layers):
        super().__init__()
        assert isinstance(mod, nn.Linear)
        self.mod = mod
        self.indep_across_layers = indep_across_layers

        self.out_features, self.in_features = self.mod.weight.shape
        self.in_features_bias = self.in_features + int(mod.bias is not None)
        if self.bias is not None:
            assert self.bias.shape == (self.out_features,)

        #Approximation to the inverse weight matrix for inverting backprop.
        self.grad_lin_reg = MultivariateLinearRegressionNoBias(mod.in_features, mod.out_features)

        def hook(mod, grad_input, grad_outputs):
            grad_lin_reg.update(grad_input, grad_output)
        self.register_full_backward_hook(hook)

        #Inverses of the Kronecker factored Cholesky of the covariance matrix.
        #chol^{-1}(cov_left), i.e. lower-triangular matrix, representing the 
        #inverse of the Cholesky of the left Kronecker factor of the covariance.
        self.register_buffer('invL', t.eye(self.out_features))
        #chol^{-T}(cov_right) i.e. upper-triangular matrix, representing the
        #transpose of the inverse of the Cholesky of the right Kronecker factor of the covariance.
        self.register_buffer('invU', t.eye(self.in_features_bias))

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
