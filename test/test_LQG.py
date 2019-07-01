#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:13:43 2019

@author: sadra
"""
import numpy as np
from pypolytrajectory.system import LQG
import pydrake.systems.controllers as PC

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