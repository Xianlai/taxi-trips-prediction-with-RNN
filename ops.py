#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script implements the tensorflow operation functions.

Author: Xian Lai
Project: NYC taxi pickups pattern learning
Date: Mar. 03, 2018
"""
from helpers import *
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.tools import inspect_checkpoint as chkp

tanh = tf.tanh
relu = tf.nn.relu
softmax = tf.nn.softmax
sigmoid = tf.sigmoid
softplus = tf.nn.softplus
norm_init = tf.random_normal_initializer
ones_init = tf.initializers.ones
zeros_init = tf.zeros_initializer
const_init = tf.constant_initializer

# ---------------------------- OPERATIONS ------------------------------------
def minmax_scale_tensor(tensor):
    """
    """
    min_ = tf.reduce_min(tensor, axis=1, keepdims=True)
    tensor = tensor - min_
    max_ = tf.reduce_max(tensor, axis=1, keepdims=True)
    tensor = tensor / (max_ + 1e-12)

    return tensor

def std_scale_tensor(tensor):
    """
    """
    mean, var = tf.nn.moments(tensor, axes=[1], keep_dims=True)

    return (tensor - mean) / (tf.sqrt(var + 1e-12))


def loss_l2(y, yhat):
    """
    """
    return tf.losses.mean_squared_error(y, yhat)

def loss_l1(y, yhat):
    """
    """
    return tf.sqrt(loss_l2(y, yhat))

def shifted_relu(t, name="shifted_relu"):
    """
    """
    with tf.variable_scope(name):
        shift = tf.get_variable(
            "shift",
            shape=[1],
            dtype=t.dtype,
            initializer=zeros_init
        )
        return tf.maximum(t + shift, 0.0)

def gate(xs, output_size, act_fn, name="linear"):
    """ concatenate, linear project and activate.
    Concatenate the input matrices and linear map(wx + b) them to a output 
    matrix with given output size.

    Args:
        xs (list): a list of tensors to be concated
    """
    dtype = xs[0].dtype

    with tf.variable_scope(name):
        x_concat = tf.concat(xs, axis=1, name="concat")
        input_size = x_concat.shape[-1]
        w = tf.get_variable(
            "w",
            shape=[input_size, output_size],
            dtype=dtype,
        )
        b = tf.get_variable(
            "b",
            shape=[output_size],
            dtype=dtype
        )
        x_proj = tf.matmul(x_concat, w, name="matmul")
        x_proj = tf.add(x_proj, b, name="add_bias")
        x_act = act_fn(x_proj, name="activate")

    return x_act


def linear(x, output_size, has_bias=True, name="linear", return_weights=False):
    """ Concatenate the input matrices and linear map(wx + b) them to a output 
    matrix with given output size.

    """
    input_size = x.shape.as_list()[-1]
    dtype = x.dtype

    with tf.variable_scope(name):
        w = tf.get_variable(
            "weights",
            shape=[input_size, output_size],
            dtype=dtype
        )
        if has_bias:
            b = tf.get_variable(
                "bias",
                shape=[output_size],
                dtype=dtype
            )
            if return_weights: return x @ w + b, w
            else: return x @ w + b
        else:
            if return_weights: return x @ w, w
            else: return x @ w


def linear_multi(x, output_sizes, has_bias=True, name="linear_multi"):
    """ linear project input tensor multiple times to achieve output shape
    """
    order = len(output_sizes)
    perm  = [-1] + list(range(order - 1))
    x = tf.reshape(x, shape=(1,)*order)
    with tf.variable_scope(name):
        for i, output_size in enumerate(output_sizes):
            x = tf.transpose(x, perm=perm)
            x = linear(x, output_size, name="linear_%d" %i, has_bias=has_bias)

    return x


def linear_2_scalar(x, name="linear_2_scalar"):
    """ linear project a 2-d tensor to a scalar
    """
    with tf.variable_scope(name):
        row = tf.transpose(linear(x, 1, name='shrink_cols')) # (1, memory_dim)
        res = tf.squeeze(linear(row, 1, name='shrink_rows')) # (1)

        return res


def cos_similarity(vector, matrix, name="cos"):
    """ compute cosine similarity of a row vector to each rows in matrix.

    Args:
        vector (tensor): a 1-d row vector tensor with shape (b, m)
        matrix (tensor): a 2-d matrix tensor with shape (n, m)
    
    Returns:
        similarity (tensor): a 1-d column vector tensor with shape (b, n)
    """
    assert vector.shape[-1] == matrix.shape[-1],\
        "number of columns of vector and matrix don't match"
    with tf.variable_scope(name):
        n = matrix.shape[0]
        vector = tf.tile(tf.expand_dims(vector, axis=1), [1, n, 1])
        norm_v = tf.nn.l2_normalize(vector, axis=2) 
        norm_m = tf.nn.l2_normalize(matrix, axis=1)
        cos = tf.reduce_sum(tf.multiply(norm_v,norm_m), axis=2)

        return cos
        # numerator = vector @ tf.transpose(matrix) # shape (b, n)
        # norm_v = tf.sqrt(tf.reduce_sum(tf.pow(vector, 2), axis=1, keepdims=True)) # shape (b, 1)
        # norm_m = tf.sqrt(tf.reduce_sum(tf.pow(matrix, 2), axis=1, keepdims=True)) # shape (n, 1)
        # denominator = norm_v @ tf.transpose(norm_m) + 1e-3 # shape (b, n)
    
        # return numerator / denominator # shape (b, n)


def circular_convolute(vector, kernel, name="convolution"):
    """ computes circular convolution

    Args:
        vector (tensor): a 3-D tensor (batch_size, memory_size)
        kernel (tensor): a 3-D tensor (batch_size, shift_range)

    Returns:
        (batch_size, memory_size)
    """
    with tf.variable_scope(name):
        vector_size  = int(vector.get_shape()[-1])
        kernel_shift = int(math.floor(int(kernel.get_shape()[-1])/2.0))
        kernel_range = range(kernel_shift, -kernel_shift - 1, -1) 

        kernels = [] # a vector_size-long list with tensors with shape (batch_size, 1)
        for i in range(vector_size):
            indices = [wrap_index(i+j, vector_size) for j in kernel_range]
            slice_  = tf.gather(vector, indices, axis=1) # (batch_size, kernel_size)
            integral = tf.reduce_sum(slice_ * kernel, axis=1)
            kernels.append(integral)

        convoluted = tf.dynamic_stitch([i for i in range(vector_size)], kernels)

        return tf.transpose(convoluted)



def get_variables(scope="RNN"):
    """
    """
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def print_shape(tensor):
    """
    """
    print(tensor.name, tensor.shape.as_list())



