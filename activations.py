from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import warnings
import keras
from keras import backend as K
#import tensorflow as K
import numpy as np
import tensorflow as tf # USING TF HERE LEADS TO STRANGE ERRORS!
from keras.utils.generic_utils import deserialize_keras_object
from keras.engine import Layer

def exp(x, eps=None):
    """
    Exponential unit.
    NOTE: This can return inf for large inputs!
    """ 
    if eps is None:
        eps = 0.0
    act = lambda x: K.exp(x) + eps
    return act(x)


def el(x, eps=None):
    """
    Exponential linear unit.
    """ 
    if eps is None:
        eps = 0.0
    act = lambda x: K.elu(x) + 1. + eps
    return act(x)


def elog(x, eps=None):
    """
    Exponential log unit.
    Uses TF workaround from: https://github.com/tensorflow/tensorflow/issues/2540
    """ 
    if eps is None:
        eps = 0.0
    exp_safe = tf.where(x <= 0, K.exp(x), 1.)
    a = 1.7632228343518967
    log_safe = tf.where(x < 0, 0,  a * tf.log(x + a))
    act = lambda x: tf.where(x <= 0, exp_safe, log_safe) + eps
    return act(x)

def et(x, eps=None, max_val=2.):
    """
    Exponential tanh unit.

    Args:
        x:

    """
    if eps is None:
        eps = 0.0
    exp_safe = tf.where(x <= 0, K.exp(x), tf.ones_like(x))
    a = max_val - 1.
    act = lambda x: tf.where(x < 0, exp_safe, a * tf.keras.backend.tanh(x/a) + 1) + eps
    return act(x)
 
def il(x, eps=None):
    """
    Approaches 0 for x-> -inf as 1/(1-x), with f(0)=1 and f'(0)=1. Cont differentiable.
    
    Uses TF workaround from: https://github.com/tensorflow/tensorflow/issues/2540
    
    Args:
        x:

    """
    if eps is None:
        eps = 0.0
    inv_safe = tf.where(x < 0, 1./(1. - x), tf.ones_like(x))
    act = lambda x: tf.where(x < 0, inv_safe, x+1) + eps
    return act(x)
 
def ilog(x, eps=None):
    """
    
    Args:
        x:

    Note:
        Tensorflow where statement will evaluate all cases, which can lead to instabilities?
    """
    if eps is None:
        eps = 0.0
    inv_safe = tf.where(x < 0, 1./(1. - x), tf.ones_like(x))
    a = 1.7632228343518967
    log_safe = tf.where(x < 0, 0,  a * tf.log(x + a))
    act = lambda x: tf.where(x < 0, inv_safe, log_safe) + eps
    return act(x)
 
def it(x, eps=None, max_val=2.):
    """
    
    Args:
        x:

    """
    a = max_val - 1.
    if eps is None:
        eps = 0.0
    inv_safe = tf.where(x < 0, 1./(1. - x), tf.ones_like(x))
    act = lambda x: tf.where(x < 0, inv_safe, a * tf.keras.backend.tanh(x/a) + 1) + eps
    return act(x)
