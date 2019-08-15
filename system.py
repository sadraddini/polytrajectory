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

# Pypolycontain
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.zonotope_order_reduction.methods import Girard


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
        self.W={} # Zonotopes for process disturbance
        self.C={} # C_t
        self.V={} # Zonotopes
        self.D={} # Performance Output
        self.d={} # Performance Offset
        self.RR={} # R_t
        self.QQ={} # Q_t
        self.E={} # E_t dynamics error
        self.F={} # F_t estimation error
        self.M,self.N,self.R,self.S={},{},{},{}
        self.Xi={} # The zonotope of xi
        # LQG
        self.L=[None]
        self.K=[None]
        
        
    def construct_dimensions(self):
        self.n,self.m,self.o,self.z=self.B[0].shape[0],self.B[0].shape[1],self.C[0].shape[0],self.D[0].shape[0]
    
    def _abstract_evolution_matrices(self,T=0):
        self.P={}
        if T==0:
            T=max(self.A.keys())+1
        # Construct P_x
        self.P["x",0]=np.eye(self.n)
        self.P["x",1]=self.A[0]
        for t in range(1,T):
            self.P["x",t+1]=np.dot(self.A[t],self.P["x",t])
        # Construct P_u
        self.P["u",0]=np.zeros((self.n,0))
        self.P["u",1]=self.B[0]
        for t in range(1,T):
            self.P["u",t+1]=np.hstack( (np.dot(self.A[t],self.P["u",t]),self.B[t] ))
        # Construct P_w
        self.P["w",0]=np.zeros((self.n,0))
        self.P["w",1]=np.eye(self.n)
        for t in range(1,T):
            self.P["w",t+1]=np.hstack(( np.dot(self.A[t],self.P["w",t]),np.eye(self.n) )) 
        self.Q={}
        self.Q["x",0]=self.C[0]
        for t in range(T):
            self.Q["x",t+1]=np.vstack((self.Q["x",t],np.dot(self.C[t],self.P["x",t+1])))
        self.Q["u",0]=np.zeros((self.o,self.m))
        self.Q["w",0]=np.zeros((self.o,self.n))
        for t in range(T):
            self.Q["u",t+1]=triangular_stack(self.Q["u",t],\
                    np.hstack(( np.dot(self.C[t],self.P["u",t+1]),np.zeros((self.o,self.m)) )) )
            self.Q["w",t+1]=triangular_stack(self.Q["w",t],\
                    np.hstack(( np.dot(self.C[t],self.P["w",t+1]),np.zeros((self.o,self.n)) )) )
        for t in range(T):
            self.Q["v",t]=np.eye(self.o*(t+1))
            
    def _error_zonotopes(self,T=0):
        Sigma={}
        Psi={}
        if T==0:
            T=max(self.A.keys())
        # Construction of Xi:
        self.Xi,self.Xi_reduced,self.F_reduced={},{},{}
        self.Xi["x",0]=self.C[0]
        self.Xi["w",0]=np.zeros((self.o,self.n))
        self.Xi["v",0]=np.eye(self.o)
        for t in range(T):
#            print "t=",t
            Gw=spa.block_diag(*[self.W[t].G for tau in range(0,t+1)])
            Gv=spa.block_diag(*[self.V[t].G for tau in range(0,t+1)])
            Wbar=np.vstack([self.W[tau].x for tau in range(0,t+1)])
            Vbar=np.vstack([self.V[tau].x for tau in range(0,t+1)])
            L1=np.linalg.multi_dot([self.C[t+1],self.P["x",t+1],self.X0.G])
            L2=np.linalg.multi_dot([self.C[t+1],self.P["w",t+1],Gw])
            L3=np.zeros((self.o,Gv.shape[1]))
            L4=self.V[t+1].G
            F1=np.dot(self.Q["x",t],self.X0.G)
            F2=np.dot(self.Q["w",t],Gw)
            F3=np.dot(self.Q["v",t],Gv)
            F4=np.zeros(((t+1)*self.o,self.V[t+1].G.shape[1]))
            Sigma["e",t]=np.hstack((L1,L2,L3,L4))
            Psi["e",t]=np.hstack((F1,F2,F3,F4))
#            print Sigma["e",t].shape
#            print Psi["e",t].shape
            K_1=np.linalg.multi_dot([self.D[t],self.P["x",t],self.X0.G])
            K_2=np.dot(np.hstack((np.dot(self.D[t],self.P["w",t]),np.zeros((self.z,self.n)))),Gw)
            K_3=np.zeros((self.z,Gv.shape[1]))
            P_1=np.dot(self.Q["x",t],self.X0.G)
            P_2=np.dot(self.Q["w",t],Gw)
            P_3=-np.dot(self.Q["v",t],Gv)
            Sigma["f",t]=np.hstack((K_1,K_2,K_3))
            Psi["f",t]=np.hstack((P_1,P_2,P_3))
#            print "f",Sigma["f",t].shape
#            print "f",Psi["f",t].shape   
            self.M[t]=np.dot(Sigma["e",t],np.linalg.pinv(Psi["e",t]))
            self.R[t]=np.dot(Sigma["f",t],np.linalg.pinv(Psi["f",t]))
            self.N[t]=np.dot(self.C[t+1],self.P["u",t+1])-np.dot(self.M[t],self.Q["u",t])
            self.S[t]=np.hstack(( np.dot(self.D[t],self.P["u",t]),np.zeros((self.z,self.m)) ))-np.dot(self.R[t],self.Q["u",t])
#            print self.M[t].shape
#            print self.N[t].shape
#            print self.R[t].shape
#            print self.S[t].shape
            ebar=np.linalg.multi_dot([self.C[t+1],self.P["x",t+1],self.X0.x])\
                    -np.linalg.multi_dot([self.M[t],self.Q["x",t],self.X0.x])\
                    +np.linalg.multi_dot([self.C[t+1],self.P["w",t+1],Wbar])\
                    -np.linalg.multi_dot([self.M[t],self.Q["w",t],Wbar])\
                    +np.linalg.multi_dot([self.M[t],self.Q["v",t],Vbar])\
                    +self.V[t+1].x
            eG=Sigma["e",t]-np.dot(self.M[t],Psi["e",t])
            self.E[t]=zonotope(ebar,eG)               
            fbar=np.linalg.multi_dot([self.D[t],self.P["x",t],self.X0.x])\
                    -np.linalg.multi_dot([self.R[t],self.Q["x",t],self.X0.x])\
                    +np.dot(np.hstack((np.dot(self.D[t],self.P["w",t]),np.zeros((self.z,self.n)))),Wbar)\
                    -np.linalg.multi_dot([self.R[t],self.Q["w",t],Wbar])\
                    -np.linalg.multi_dot([self.R[t],self.Q["v",t],Vbar])\
                    +self.d[t]
            fG=Sigma["f",t]-np.dot(self.R[t],Psi["f",t])
            self.F[t]=zonotope(fbar,fG)
            F_G_reduced=Girard(fG,fG.shape[0])
            self.F_reduced[t]=zonotope(fbar,F_G_reduced)
        for t in range(T):
            self.Xi["x",t+1]=triangular_stack(self.Xi["x",t],np.dot(self.C[t+1],self.P["x",t+1])-np.dot(self.M[t],self.Q["x",t]))
            self.Xi["w",t+1]=triangular_stack(self.Xi["w",t],np.hstack((np.dot(self.C[t+1],self.P["w",t+1])-np.dot(self.M[t],self.Q["w",t]),\
              np.zeros((self.o,self.n)) )) )
            self.Xi["v",t+1]=triangular_stack(self.Xi["v",t],np.hstack((np.dot(self.M[t],self.Q["v",t]),np.eye(self.o))))
        for t in range(T):
            Gw=spa.block_diag(*[self.W[t].G for tau in range(0,t+1)])
            Gv=spa.block_diag(*[self.V[t].G for tau in range(0,t+1)])
            Wbar=np.vstack([self.W[tau].x for tau in range(0,t+1)])
            Vbar=np.vstack([self.V[tau].x for tau in range(0,t+1)])
#            print "t=",t,"Xi_shapes",Xi["x",t].shape,Xi["w",t].shape,Xi["v",t].shape
#            print "bar",Wbar.shape,Vbar.shape
            Xi_bar=np.dot(self.Xi["x",t],self.X0.x)+np.dot(self.Xi["w",t],Wbar)+np.dot(self.Xi["v",t],Vbar)
            Xi_G=np.hstack((np.dot(self.Xi["x",t],self.X0.G),np.dot(self.Xi["w",t],Gw),np.dot(self.Xi["v",t],Gv)))
            self.Xi[t]=zonotope(Xi_bar,Xi_G)
            Xi_G_reduced=Girard(Xi_G,Xi_G.shape[0])
            self.Xi_reduced[t]=zonotope(Xi_bar,Xi_G_reduced)
            
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
    L=np.linalg.multi_dot([P,C.T,np.linalg.inv(Y)])
    return L,K

def LQG_LTV(sys,T):
    """
    Linear Time-Varying Linear Quadratic Guassian Control
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
        alpha=np.linalg.multi_dot([sys.B[t].T,S[t+1],sys.B[t]])+sys.RR[t]
        beta=S[t+1]-np.linalg.multi_dot([S[t+1],sys.B[t],np.linalg.inv(alpha),sys.B[t].T,S[t+1]])
        S[t]=np.linalg.multi_dot([sys.A[t].T,beta,sys.A[t]])+sys.QQ[t]
    for t in range(T):
        X=np.linalg.multi_dot([sys.B[t].T,S[t+1],sys.B[t]])+sys.RR[t]
        K[t]=np.linalg.multi_dot([np.linalg.inv(X),sys.B[t].T,S[t+1],sys.A[t]])
    for t in range(T+1):
        Y=np.linalg.multi_dot([sys.C[t],P[t],sys.C[t].T])+sys.V[t].G
#        L[t]=np.linalg.multi_dot([sys.A[t],P[t],sys.C[t].T,np.linalg.inv(Y)])
        L[t]=np.linalg.multi_dot([P[t],sys.C[t].T,np.linalg.inv(Y)])
    return L,K    

def triangular_stack(A,B):
    q=B.shape[1]-A.shape[1]
    assert q>=0
    return np.vstack((np.hstack((A,np.zeros((A.shape[0],q)))),B))