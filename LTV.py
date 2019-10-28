#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:50:02 2019

@author: sadra
"""

"""
LTV systems in the form of:
    x_{t+1} = A_t x_t + B_t u_t + w_t
       y_t  = C_t x_t + v_t

We assume \W_t and \V_t are given polytopes (zonotopes)
"""

import numpy as np
import scipy.linalg as spa

class system:
    def __init__(self):
        self.A={} # A_t
        self.B={} # B_t
        self.W={} # Zonotopes
        self.C={} # C_t
        self.V={} # Zonotopes
        
    def construct_dimensions(self):
        self.n,self.m,self.o=self.B[0].shape[0],self.B[0].shape[1],self.C[0].shape[0]
    
    def construct_E(self,T=0):
        self.E={}
        if T==0:
            T=max(self.A.keys())+1
        # Construct E_x
        self.E["x",0]=self.A[0]
        for t in range(1,T):
            self.E["x",t]=np.dot(self.A[t],self.E["x",t-1])
        # Construct E_u
        self.E["u",0]=self.B[0]
        for t in range(1,T):
            self.E["u",t]=np.hstack( (np.dot(self.A[t],self.E["u",t-1]),self.B[t] ))
        # Construct E_u
        self.E["w",0]=np.eye(self.n)
        for t in range(1,T):
            self.E["w",t]=np.hstack(( np.dot(self.A[t],self.E["w",t-1]),np.eye(self.n) )) 
        self.F={}
        for t in range(T):
            self.F["x",t]=np.vstack(([self.C[0]]+[np.dot(self.C[tau],self.E["x",tau-1]) for tau in range(1,t+1)] ))
        self.F["u",0]=np.zeros((self.o,self.m))
        self.F["w",0]=np.zeros((self.o,self.n))
        for t in range(1,T):
            self.F["u",t]=np.vstack(( np.hstack((self.F["u",t-1], np.zeros((self.o*(t),self.m)))),\
                    np.hstack(( np.dot(self.C[t],self.E["u",t-1]),np.zeros((self.o,self.m)) )) ))
            self.F["w",t]=np.vstack(( np.hstack((self.F["w",t-1], np.zeros((self.o*(t),self.n)))),\
                    np.hstack(( np.dot(self.C[t],self.E["w",t-1]),np.zeros((self.o,self.n)) )) ))
        for t in range(T):
            self.F["v",t]=spa.block_diag(*[0*np.eye(self.o*t),np.eye(self.o)])
            
def test_controllability(A,B):
    n=A.shape[0]
    C=np.hstack([np.dot(np.linalg.matrix_power(A,i),B) for i in range(n)])
    r=np.linalg.matrix_rank(C)
    print(r)
    if r==n:
        print("system is controllable as %d = %d"%(r,n))
    else:
        print("system is NOT controllable as %d < %d"%(r,n))
