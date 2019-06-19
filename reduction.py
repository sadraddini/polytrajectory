#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:13:54 2019

@author: sadra
"""
# Numpy ans scipy
import numpy as np
import scipy.linalg as spa
# pydrake
import pydrake.solvers.mathematicalprogram as MP
import pydrake.solvers.gurobi as Gurobi_drake
import pydrake.solvers.osqp as OSQP_drake
# Pypolycontain
from pypolycontain.lib.objects import zonotope
# use Gurobi solver
global gurobi_solver,OSQP_solver
gurobi_solver=Gurobi_drake.GurobiSolver()
OSQP_solver=OSQP_drake.OsqpSolver()


def reduced_order(sys,T):
    M,N,Z={},{},{}
    for t in range(T):
        # 1: intiial state
        K_1=np.dot(sys.F["x",t],sys.X0.G)
        e_1=np.dot(sys.C[t+1],sys.E["x",t])
        L_1=np.dot(e_1,sys.X0.G)
        # 2: process noice
        K_2=np.dot(sys.F["w",t],spa.block_diag(*[sys.W[t].G for tau in range(0,t+1)]))
        e_2=np.dot(sys.C[t+1],sys.E["w",t])
        L_2=np.dot(e_2,spa.block_diag(*[sys.W[t].G for tau in range(0,t+1)]))
        # 3: observation noise
        K_3=np.dot(sys.F["v",t],spa.block_diag(*[sys.V[t].G for tau in range(0,t+1)]))
        L_3=np.zeros((sys.o,K_3.shape[1]))
        # Build the L and K matrices: L=M*K
        L=np.hstack((L_1,L_2,L_3))
        K=np.hstack((K_1,K_2,K_3))
        M[t]=np.dot(L,np.linalg.pinv(K))
        N[t]=np.dot(sys.C[t+1],sys.E["u",t])-np.dot(M[t],sys.F["u",t])
        Z[t]=zonotope(None,None)
        Z[t].G=np.hstack((L-np.dot(M[t],K),sys.V[t+1].G))
        Z[t].x=np.dot(e_1-np.dot(M[t],sys.F["x",t]),sys.X0.x)\
            +np.dot(e_2-np.dot(M[t],sys.F["w",t]),np.vstack([sys.W[t].x for tau in range(0,t+1)]))\
            +np.dot(np.dot(M[t],sys.F["v",t]),np.vstack([sys.V[t].x for tau in range(0,t+1)]))\
            +sys.V[t+1].x
        print "error in initial condition:",e_1-np.dot(M[t],sys.F["x",t])
    return M,N,Z

