#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:13:43 2019

@author: sadra
"""
import numpy as np
from pypolytrajectory.system import LQG,LQG_LTV,LTV

import pydrake.systems.controllers as PC

from pypolycontain.lib.objects import zonotope


def test_LQG():
    n=5
    m=2
    o=3
    np.random.seed(5)
    A=2*(np.random.random((n,n))-0.5)
    B=np.random.random((n,m))
    C=np.random.random((o,n))
    W=np.eye(n)*1
    V=np.eye(o)*1
    Q=np.eye(n)*1
    R=np.eye(m)*1
    
    L,K=LQG(A,B,C,W,V,Q,R)
    A_cl=A-np.dot(B,K)
    A_observer=A-np.dot(L,C)
    print "open_loop control:",np.linalg.eigvals(A)
    print "clsoed_loop control:",np.linalg.eigvals(A_cl)
    print "Observer:",np.linalg.eigvals(A_observer)
    
    print "\n"*3
    
    K,J_c=PC.DiscreteTimeLinearQuadraticRegulator(A,B,1*np.eye(n),np.eye(m))
    L,J_o=PC.DiscreteTimeLinearQuadraticRegulator(A.T,C.T,1*np.eye(n),np.eye(o))
    
    A_cl=A-np.dot(B,K)
    A_observer=A-np.dot(L.T,C)
    print "open_loop control:",np.linalg.eigvals(A)
    print "clsoed_loop control:",np.linalg.eigvals(A_cl)
    print "Observer:",np.linalg.eigvals(A_observer)
    
#test_LQG()
    
def test_LQG_LTV():
    S=LTV()
    print S.R
    n=7
    m=1
    o=1
    T=55
    np.random.seed(212)
    S.X0=zonotope(np.ones((n,1))*0,np.eye(n)*1)
    for t in range(T):
        S.A[t]=0.95*np.eye(n)+np.random.normal(size=(n,n))*0.01
        S.B[t]=np.random.randint(0,2,size=(n,m))
        S.C[t]=np.zeros((o,n))
        S.C[t][0,0]=1
        S.W[t]=zonotope(np.zeros((n,1)),np.eye(n)*0.01)
        S.V[t]=zonotope(np.zeros((o,1)),np.eye(o)*0.01)
        S.Q[t]=np.eye(n)*1
        S.R[t]=np.eye(m)*1
        S.F=np.eye(n)*1
    L,K=LQG_LTV(S,T=50)

test_LQG_LTV()