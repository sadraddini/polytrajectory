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
from pypolycontain.lib.zonotope_order_reduction.methods import G_cut,Girard
# use Gurobi solver
global gurobi_solver,OSQP_solver
gurobi_solver=Gurobi_drake.GurobiSolver()
OSQP_solver=OSQP_drake.OsqpSolver()





def reduced_order_new(sys,T):
    M,N,Z={},{},{}
    for t in range(T):
        # 1: intiial state
        Omega_1=np.dot(sys.Q["x",t],sys.X0.G)
        e_1=np.dot(sys.C[t+1],sys.P["x",t+1])
        Gamma_1=np.dot(e_1,sys.X0.G)
        gamma_1=np.dot(e_1,sys.X0.x)
        # 2: process noice
        K_2=np.dot(sys.Q["w",t],spa.block_diag(*[sys.W[t].G for tau in range(0,t+1)]))
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
        Z["X0",t]=L_1-np.dot(M[t],K_1)
        Z["W",t]=L_2-np.dot(M[t],K_2)
        Z["V",t]=np.hstack((L_3-np.dot(M[t],K_3),sys.V[t+1].G))
    return M,N,Z


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
        Z["X0",t]=L_1-np.dot(M[t],K_1)
        Z["W",t]=L_2-np.dot(M[t],K_2)
        Z["V",t]=np.hstack((L_3-np.dot(M[t],K_3),sys.V[t+1].G))
    return M,N,Z

def error_construction_old(sys,T,q0):
    sys.ebar={}
    for t in range(-1,T+1):
        sys.E[t]=zonotope(None,None)
    sys.E[-1].x=np.zeros((q0,1))
    for t in range(T+1):
        sys.ebar[t]=sys.Z[t].x
    # t=0
    sys.E[-1].G=np.eye(q0)
    sys.ebar[-1]=sys.E[-1].x
    # Other t
    for t in range(T+1):
        sys.E[t].x=np.vstack([sys.ebar[tau] for tau in range(-1,t+1)])
        sys.E[t].G=triangular_stack(sys.E[t-1].G,sys.Z[t].G)

def error_construction(sys,T,q0):
    sys.ebar={}
    for t in range(-1,T+1):
        sys.E[t]=zonotope(None,None)
    sys.E[-1].x=np.zeros((q0,1))
    for t in range(T+1):
        sys.ebar[t]=sys.Z[t].x
    # t=0
    sys.E[-1].G=np.eye(q0)
    sys.ebar[-1]=sys.E[-1].x
    for source in ["X0","V","W"]:
        sys.E[source,-1,"G"]=np.zeros((0,0))
    # Other t
    for t in range(T+1):
        sys.E[t].x=np.vstack([sys.ebar[tau] for tau in range(-1,t+1)])
        for source in ["X0","V","W"]:
            sys.E[source,t,"G"]=triangular_stack(sys.E[source,t-1,"G"],sys.Z[source,t])
        A=np.hstack([sys.E[source,t,"G"] for source in ["X0","V","W"] ])
        sys.E[t].G=triangular_stack(sys.E[-1].G,A)
    print "All sources complete"
    

def order_reduction_error(sys,T,q=0):
    # Order reduection
    for t in range(T+1):
        sys.E[t].G=Girard(sys.E[t].G,q+sys.E[t].G.shape[0])
    

def triangular_stack(A,B):
    q=B.shape[1]-A.shape[1]
    assert q>=0
    return np.vstack((np.hstack((A,np.zeros((A.shape[0],q)))),B))    