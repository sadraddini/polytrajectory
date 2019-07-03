#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:24:55 2019

@author: sadra
"""

# Numpy and Scipy
import numpy as np
import scipy.linalg as spa

# Pydrake
from pydrake.math import DiscreteAlgebraicRiccatiEquation as DARE


class LTI:
    """
    LTV systems in the form of:
        x_{t+1} = A x_t + B u_t + w_t
           y_t  = C x_t + v_t
    
    We assume \W and \V are given polytopes (zonotopes), or covariance matrix for normal distributions
    """
    def __init__(self,A=None,B=None,C=None,W=None,V=None):
        self.A=A
        self.B=B
        self.W=W # Zonotopes
        self.C=C # C_t
        self.V=V # Zonotopes
        
    def construct_dimensions(self):
        self.n,self.m,self.o=self.B.shape[0],self.B.shape[1],self.C.shape[0]
    
    def LQG(self,Q,R):
        """
        We solve the following Riccati Equation:
            A′XA−X−A′XB(B′XB+R)^−1 B′XA+Q=0
            Then we have:
                K=(B'SB+R)^-1 B'SA
                L=PC'(CPC'+W)^-1
        """
        return LQG(self.A,self.B,self.C,self.W,self.V,self.Q,self.R)
    
class LTV:
    def __init__(self):
        self.A={} # A_t
        self.B={} # B_t
        self.W={} # Zonotopes
        self.C={} # C_t
        self.V={} # Zonotopes
        self.R={} # R_t
        self.Q={} # Q_t
        self.F=None
        
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
            
        
def LQG(A,B,C,W,V,Q,R):
    """
    The optimal infinite-time linear quadratic regulator.
    The euqations are here:
        https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic%E2%80%93Gaussian_control
    """
    P=DARE(A=A.T,B=C.T,Q=W,R=V)
    S=DARE(A=A,B=B,Q=Q,R=R)
    X=np.linalg.multi_dot([B.T,S,B])+R
    K=np.linalg.multi_dot([np.linalg.inv(X),B.T,S,A])
    Y=np.linalg.multi_dot([C,P,C.T])+V
    L=np.linalg.multi_dot([A,P,C.T,np.linalg.inv(Y)])
    return L,K

def LQG_LTV(sys,T):
    """
    Linear Time-Varying Linear Quadratic Guassiang Control
    We solve the Riccati difference equations
    """
    P,S,L,K={},{},{},{}
    P[0]=sys.X0.G
    S[T]=sys.F_cost
    for t in range(T+1):
        alpha=np.linalg.multi_dot([sys.C[t],P[t],sys.C[t].T])+sys.V[t].G
        beta=P[t]-np.linalg.multi_dot([P[t],sys.C[t].T,np.linalg.inv(alpha),sys.C[t],P[t]])
        P[t+1]=np.linalg.multi_dot([sys.A[t],beta,sys.A[t].T])+sys.W[t].G
    for t in range(T-1,-1,-1):
        alpha=np.linalg.multi_dot([sys.B[t].T,S[t+1],sys.B[t]])+sys.R[t]
        beta=S[t+1]-np.linalg.multi_dot([S[t+1],sys.B[t],np.linalg.inv(alpha),sys.B[t].T,S[t+1]])
        S[t]=np.linalg.multi_dot([sys.A[t].T,beta,sys.A[t]])+sys.Q[t]
    for t in range(T+1):
        X=np.linalg.multi_dot([sys.B[t].T,S[t],sys.B[t]])+sys.R[t]
        K[t]=np.linalg.multi_dot([np.linalg.inv(X),sys.B[t].T,S[t],sys.A[t]])
        Y=np.linalg.multi_dot([sys.C[t],P[t],sys.C[t].T])+sys.V[t].G
        L[t]=np.linalg.multi_dot([sys.A[t],P[t],sys.C[t].T,np.linalg.inv(Y)])
    return L,K    