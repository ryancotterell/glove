#!/usr/bin/env python
# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: c_string_type=unicode, c_string_encoding=ascii
#cython: profile=False
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = ["-std=c++11"]

# Based on code by Jonathn Raiman
# Contains substantial variable renamings to accomodate my weltanschauung

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp, log, pow, sqrt

ctypedef np.float64_t REAL_t
ctypedef np.uint32_t  INT_t

cdef inline void gaussian(
        REAL_t* W, REAL_t* C,
        REAL_t* gradsqW, REAL_t* gradsqC,
        REAL_t* error,
        int vector_size, REAL_t step_size,
        long l1, long l2, REAL_t dot, REAL_t x_ij, REAL_t f_ij) nogil:
    
    """ Gaussian ``emission'' distribution """
    cdef REAL_t gW, gC
    cdef int b
    cdef REAL_t diff = dot - log(x_ij)
    cdef REAL_t fdiff = f_ij * diff
    
    
    # weighted squared error
    error[0] += 0.5 * fdiff * diff
    
    # multiple in step size
    fdiff *= step_size

    # take gradient steps with AdaGrad
    for b in range(vector_size):
        # compute the gradients 
        gW = fdiff * C[b + l2]
        gC = fdiff * W[b + l1]
        
        # adaptive learning rate updates (Duchi et al. 2011)
        W[b + l1] -= (gW / sqrt(gradsqW[b + l1]))
        C[b + l2] -= (gC / sqrt(gradsqC[b + l2]))
        gradsqW[b + l1] += gW * gW
        gradsqC[b + l2] += gC * gC

cdef inline void poisson(
        REAL_t* W, REAL_t* C,
        REAL_t* gradsqW, REAL_t* gradsqC,
        REAL_t* error,
        int vector_size, REAL_t step_size,
        long l1, long l2, REAL_t dot, REAL_t x_ij, REAL_t f_ij) nogil:
    """ Poisson ``emission'' distribution """
    cdef REAL_t gW, gC
    cdef int b
    cdef REAL_t score = -x_ij * dot + exp(dot)
    cdef REAL_t fscore = f_ij * score
    
    # weighted squared error
    error[0] += fscore * score
    
    # multiple in step size
    fscore *= step_size

    # take gradient steps with AdaGrad
    for b in range(vector_size):
        # compute the gradients 
        gW = (fscore + exp(dot)) * C[b + l2]
        gC = (fscore + exp(dot)) * W[b + l1]
        
        # adaptive learning rate updates (Duchi et al. 2011)
        W[b + l1] -= (gW / sqrt(gradsqW[b + l1]))
        C[b + l2] -= (gC / sqrt(gradsqC[b + l2]))
        gradsqW[b + l1] += gW * gW
        gradsqC[b + l2] += gC * gC

cdef void train_thread(
        REAL_t* W, REAL_t* C,
        REAL_t* gradsqW, REAL_t* gradsqC,
        REAL_t* error,
        INT_t* words, INT_t* contexts, REAL_t* target,
        int vector_size, int batch_size, REAL_t x_max, REAL_t alpha, REAL_t step_size) nogil:

    cdef long long a, b, l1, l2
    cdef int example_idx = 0
    cdef REAL_t gW, gC, dot, score, x_ij, f_ij

    for example_idx in range(batch_size):
        # calculate cost, save diff for gradients
        l1 = words[example_idx]    * vector_size # cr word indices start at 1
        l2 = contexts[example_idx] * vector_size

        # dot product of word and context word vector
        dot = 0.0
        for b in range(vector_size):
            dot += W[b + l1] * C[b + l2]

        # get matrix entry
        x_ij = target[example_idx]

        # compute annealing term
        f_ij = 1.0 if (x_ij > x_max) else pow(x_ij / x_max, alpha)
        gaussian(W, C, gradsqW, gradsqC, error, vector_size, step_size, l1, l2, dot, x_ij, f_ij)


def train_model(model, jobs, float _step_size, _error):
    cdef REAL_t* W = <REAL_t*>(np.PyArray_DATA(model.W))
    cdef REAL_t* C = <REAL_t*>(np.PyArray_DATA(model.C))
    cdef REAL_t* gradsqW = <REAL_t*>(np.PyArray_DATA(model.gradsqW))
    cdef REAL_t* gradsqC = <REAL_t*>(np.PyArray_DATA(model.gradsqC))

    cdef REAL_t* error = <REAL_t*>(np.PyArray_DATA(_error))

    cdef INT_t* words = <INT_t*>(np.PyArray_DATA(jobs[0]))
    cdef INT_t* contexts = <INT_t*>(np.PyArray_DATA(jobs[1]))
    cdef REAL_t* target = <REAL_t*>(np.PyArray_DATA(jobs[2]))

    # configuration and parameters
    cdef REAL_t step_size = _step_size
    cdef int vector_size = model.d
    cdef int batch_size = len(jobs[0])
    cdef REAL_t x_max  = model.x_max
    cdef REAL_t alpha  = model.alpha

    # release GIL & train on the sentence
    with nogil:
        train_thread(
            W,\
            C,\
            gradsqW,\
            gradsqC,\
            error,\
            words,\
            contexts,\
            target, \
            vector_size,\
            batch_size, \
            x_max, \
            alpha, \
            step_size
        )
