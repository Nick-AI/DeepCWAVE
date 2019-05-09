# Implementation of Beta or Dirichlet output objective. 
# Use this for custom gradients. https://www.tensorflow.org/api_docs/python/tf/custom_gradient
# https://groups.google.com/forum/#!searchin/keras-users/output$20distribution/keras-users/caIEicDknlU/ut7nch8bDAAJ
import numpy as np
#from keras import backend as K
import tensorflow as K # Keras.backend doesn't have multiply, lgamma, etc.

def DirichletMultinomial_NLL(y_true, y_pred, ignore_n=False):
    '''
    Compute NLL for Dirichlet output layer.

    Args:
        y_true: NxK matrix of (data) target values. Each row sums to 1.  
        y_pred: NxK matrix of parameters. Each row parameterizes a Dirichlet. 

    Returns:
        nll: Length N vector of the negative log likelihood.

    Note: To fully specify the distribution we need to specify a 
          distribution over the number of trials. Here we avoid 
          modelling the number of trials, and assume it is provided
          with each example.
    '''

    alpha = y_pred
    t1  = K.reduce_sum(K.lgamma(alpha + y_true), axis=1) 
    t2  = - K.lgamma(K.reduce_sum(alpha + y_true, axis=1))
    t3  = K.lgamma(K.reduce_sum(alpha, axis=1))
    t4  = - K.reduce_sum(K.lgamma(alpha), axis=1) 
    nll = - (t1 + t2 + t3 + t4)
    if not ignore_n:
        # Divide per example loss by number of trials in that example.
        # Certainly makes loss more interpretable.
        nll = nll / K.reduce_sum(y_true, axis=-1)
    return nll

def DirichletMultinomial_Mean_MSE(y_true, y_pred):
    '''
    Args:
        y_true: NxK matrix of (data) target values. Each row sums to 1.  
        y_pred: NxK matrix of parameters. Each row parameterizes a Dirichlet. 

    '''
    alpha = y_pred
    yhat   = alpha / K.expand_dims(K.reduce_sum(alpha, axis=1), 1) # MP point estimate.
    that = y_true / K.expand_dims(K.reduce_sum(y_true, axis=1), 1) # Empirical mean.
    rval   = K.reduce_mean(K.square(that - yhat), axis=-1)
    return rval



def Dirichlet_NLL(y_true, y_pred, stability_factor=1e-6):
    '''
    Compute NLL for Dirichlet output layer.

    Args:
        y_true: NxK matrix of (data) target values. Each row sums to 1.  
        y_pred: NxK matrix of parameters. Each row parameterizes a Dirichlet.
               All values should be greater than or equal to 0.

    Returns:
        nll: Length N vector of the negative log likelihood.

    Note: (1) The tensorflow log-gamma function will fail for large and small values.
          Specifically, tf.lgamma(1e-46) = inf, and tf.lgamma(1e+36)=inf.
          This is best handled by the activation function for the predictions.
          Using exponentials in either direction (e.g. exp(x) or ELU) will lead to nans.
          Try the Dirichlet-Ready Unit, with 1/(1-x) for x<0 and x+1 for x>=0.
        
          (2) The clip function kills the gradient, so only use it on the targets.

            (3) Idea: cap the loss for each particular example?
    '''
    alpha  = y_pred
    y_safe = K.keras.backend.clip(y_true, stability_factor, None) # Don't use clip when gradients are needed.
    t1  = K.reduce_sum(K.multiply((alpha - 1.0), K.log(y_safe)), axis=-1)
    t2  = K.lgamma(K.reduce_sum(alpha, axis=-1))
    t3  = - K.reduce_sum(K.lgamma(alpha), axis=-1) 
    nll = - (t1 + t2 + t3)
    #rval = K.reduce_mean(nll, axis=-1)
    return nll

def Dirichlet_Mean_MSE(y_true, y_pred):
    '''
    Compute Mean Squared Error for Mean Posterior point estimates
    of Dirichlet Output layer.

    Args:
        y_true: NxK vector of target (data) values. Rows sum to 1.
        y_pred: N x K matrix of dirichlet parameters. 
    '''
    alpha  = y_pred
    yhat   = alpha / K.expand_dims(K.reduce_sum(alpha, axis=1), 1) # MP point estimate.
    rval   = K.reduce_mean(K.square(y_true - yhat), axis=-1)
    return rval

def Beta_NLL(y_true, y_pred, k=1, stability_factor=1e-6):
    '''
    Compute mean NLL for K samples from K Beta distributions in output layer.
    Use functools.partial to change k and stability_factor.

    Args:
        y_true: nxk matrix of (data) target values. n is batch size.
        y_pred: nx2k matrix of parameters. each row parameterizes 
               k beta distributions. all values should be greater 
               than or equal to 0.

    Returns:
        nll: Length N vector of the negative log likelihood.

    Example:
        from functools import partial, update_wrapper
        loss = partial(Beta_NLL, k=2)
        update_wrapper(loss, Beta_NLL) # Keeps __name__ attribute.
    '''
    a1 = y_pred[:,:k]
    a2 = y_pred[:,k:]
 
    ysafe = K.keras.backend.clip(y_true, stability_factor, 1.-stability_factor)

    term1 = K.multiply(a1 - 1., K.log(ysafe)) + K.multiply(a2 - 1., K.log(1-ysafe))
    term2 = K.lgamma(a1 + a2) - (K.lgamma(a1) + K.lgamma(a2))

    nll  = K.reduce_sum(-(term1 + term2), axis=-1) # Reduce over axis 1 (k outputs).
    #rval = K.reduce_mean(nll, axis=-1)
    return nll

def Beta_Mean_MSE(y_true, y_pred, k=1):
    '''
    Compute Mean Squared Error for Mean Posterior point estimates
    of Beta Output layer with k targets.
    Use functools.partial to change k and stability_factor.

    Args:
        y_true: nxk matrix of (data) target values. n is batch size.
        y_pred: nx2k matrix of parameters. each row parameterizes 
               k beta distributions. all values should be greater 
               than or equal to 0.
    Example:
        from functools import partial, update_wrapper
        loss = partial(Beta_MP_MSE, k=2)
        update_wrapper(loss, Beta_MP_MSE) # Keeps __name__ attribute.

    '''
    #k = y.shape[1]
    a1 = y_pred[:,:k]
    a2 = y_pred[:,k:]
    mp = a1 / (a1 + a2) # Mean posterior.
    rval  = K.reduce_mean(K.square(mp - y_true), axis=-1)
    return rval


def Gaussian_NLL(y_true, y_pred, k=1, stability_factor=1e-24):
    """
    Compute Negative Log Likilihood for Gaussian Output Layer.

    Args:
        y_true: Nxk matrix of (data) target values.
        y_pred: Nx2k matrix of parameters. Each row parameterizes
                k Gaussian distributions with (mean, std).
        stability_factor: Min value of sigma.
    Example:
        from functools import partial, update_wrapper
        loss = partial(Gaussian_NLL, k=2)
        update_wrapper(loss, Gaussian_NLL) # Keeps __name__ attribute.
    """
    means  = y_pred[:,:k]
    sigmas = y_pred[:,k:]
    sigmasafe = sigmas + stability_factor if stability_factor else sigmas
    term1  = K.log(sigmasafe) + np.log(2 * np.pi)
    term2  = K.square((means - y_true) / sigmasafe)
    nll    = (term1 + term2) / 2.
    nll    = K.reduce_sum(nll, axis=-1) # Sum NLL over outputs, as in dirichlet. 
    #rval   = K.reduce_mean(nll, axis=-1)
    return nll

def Gaussian_MSE(y_true, y_pred, k=1):
    """
    Compute Negative Log Likilihood for Gaussian Output Layer.

    Args:
        y: Nxk matrix of (data) target values.
        alpha: Nx2k matrix of parameters. Each row parameterizes
               k Gaussian distributions with (mean, std).
    """
    means  = y_pred[:,:k]
    rval   = K.reduce_mean(K.square(means - y_true), axis=-1)
    return rval

def Laplace_NLL(y_true, y_pred, k=1, stability_factor=1e-6):
    """
    Compute Negative Log Likilihood for Laplace Output Layer.

    Args:
        y_true: Nxk matrix of (data) target values.
        y_pred: Nx2k matrix of parameters. Each row parameterizes
                k Gaussian distributions with (mean, std).
        stability_factor: Min value of sigma.
    Example:
        from functools import partial, update_wrapper
        loss = partial(Gaussian_NLL, k=2)
        update_wrapper(loss, Gaussian_NLL) # Keeps __name__ attribute.
    """
    means  = y_pred[:,:k]
    sigmas = y_pred[:,k:]
    sigmasafe = sigmas + stability_factor if stability_factor else sigmas
    term1  = K.log(2 * sigmasafe)
    term2  = K.abs(means - y_true) / sigmasafe
    nll    = term1 + term2
    nll    = K.reduce_sum(nll, axis=-1) # Sum NLL over outputs, as in dirichlet. 
    #rval   = K.reduce_mean(nll, axis=-1)
    return nll

def Gamma_NLL(y_true, y_pred, d=1, stability_factor=1e-6):
    '''
    Compute mean NLL for k samples from k Beta distributions in output layer.
    Use functools.partial to change k and stability_factor.

    Args:
        y_true: nxk matrix of (data) target values. n is batch size.
        y_pred: nx2k matrix of parameters. each row parameterizes 
               k beta distributions. all values should be greater 
               than or equal to 0.

    Returns:
        nll: Length N vector of the negative log likelihood.

    Example:
        from functools import partial, update_wrapper
        loss = partial(Gamma_NLL, d=2)
        update_wrapper(loss, Gamma_NLL) # Keeps __name__ attribute.
    '''
    k     = y_pred[:,:d]
    theta = y_pred[:,d:]
    
    term1 = - K.lgamma(k) - K.multiply(k, K.log(theta))
    term2 = K.multiply(k-1, K.log(y_true))
    term3 = - y_true / theta
    nll   = - (term1 + term2 +term3) 
    nll   = K.reduce_sum(nll, axis=-1) # Reduce over axis 1 (d outputs).
    return nll

def Gamma_Mean_MSE(y_true, y_pred, d=1):
    '''
    Compute Mean Squared Error for Mean Posterior point estimates
    of Gamma Output layer parameterized by k, theta with d targets.
    Use functools.partial to change d and stability_factor.

    Args:
        y_true: nxk matrix of (data) target values. n is batch size.
        y_pred: nx2k matrix of parameters. each row parameterizes 
               d gamma distributions. all values should be greater 
               than or equal to 0.
    Example:
        from functools import partial, update_wrapper
        loss = partial(Gamma_Mean_MSE, k=2)
        update_wrapper(loss, Gamma_Mean_MSE) # Keeps __name__ attribute.

    '''
    k     = y_pred[:,:d]
    theta = y_pred[:,d:]
    mp    = k * theta # Mean of Gamma.
    rval  = K.reduce_mean(K.square(mp - y_true), axis=-1)
    return rval


