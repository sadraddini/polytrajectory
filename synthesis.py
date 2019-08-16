#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:38:23 2019

@author: sadra
"""

# Numpy ans scipy
import numpy as np
import scipy.linalg as spa
# pydrake
import pydrake.symbolic as sym
import pydrake.solvers.mathematicalprogram as MP
import pydrake.solvers.gurobi as Gurobi_drake
import pydrake.solvers.osqp as OSQP_drake
# Pypolycontain
from pypolycontain.lib.objects import zonotope,hyperbox,H_polytope
from pypolycontain.lib.containment_pydrake import subset,subset_soft
from pypolycontain.lib.zonotope_order_reduction.methods import Girard_hull,Girard
# use Gurobi solver
global gurobi_solver,OSQP_solver
gurobi_solver=Gurobi_drake.GurobiSolver()
OSQP_solver=OSQP_drake.OsqpSolver()

import time

def output_feedback_synthesis(sys,T):
    prog=MP.MathematicalProgram()
    # Add Variables
    phi,theta,Phi,Theta={},{},{},{}
    y_tilde,u_tilde={},{} 
    Y_tilde,U_tilde={},{}
    z_bar,u_bar={},{}
    Z,U={},{} # Z matrices
    # Initial Condition
    y_tilde[0]=np.zeros((sys.o,1))
    phi[0]=np.eye(sys.o)
    # Main variables
    for t in range(T):
        theta[t]=prog.NewContinuousVariables(sys.m,sys.o*(t+1),"theta%d"%t)
        u_tilde[t]=prog.NewContinuousVariables(sys.m,1,"u_tilde%d"%t)
    # Now we the dynamics
    Phi[0],Theta[0]=phi[0],theta[0]
    theta[T]=np.zeros((sys.m,sys.o*(T+1)))
    u_tilde[T]=np.zeros((sys.m,1))
    for t in range(T):
        Y_tilde[t]=np.vstack([y_tilde[tau] for tau in range(t+1)])
        U_tilde[t]=np.vstack([u_tilde[tau] for tau in range(t+1)])
        y_tilde[t+1]=np.dot(sys.M[t],Y_tilde[t])+np.dot(sys.N[t],U_tilde[t])
        phi[t+1]=np.hstack(( np.dot(sys.M[t],Phi[t])+np.dot(sys.N[t],Theta[t]),np.eye(sys.o) ))
        Phi[t+1]=triangular_stack(Phi[t],phi[t+1])
        Theta[t+1]=triangular_stack(Theta[t],theta[t+1])
    Y_tilde[T]=np.vstack([y_tilde[tau] for tau in range(T+1)])
    U_tilde[T]=np.vstack([u_tilde[tau] for tau in range(T+1)])
    # Performance variables
    for t in range(T+1):
        # Z zonotope
        print "time construction",t
        z_bar=np.linalg.multi_dot([sys.R[t],Y_tilde[t]])+\
            np.linalg.multi_dot([sys.R[t],Phi[t],sys.Xi[t].x])+\
            np.linalg.multi_dot([sys.S[t],U_tilde[t]])+\
            np.linalg.multi_dot([sys.S[t],Theta[t],sys.Xi[t].x])+\
            sys.F[t].x
        Gz1=np.linalg.multi_dot([sys.R[t],Phi[t],sys.Xi["x",t]])+\
            np.linalg.multi_dot([sys.S[t],Theta[t],sys.Xi["x",t]])+\
            np.dot(sys.D[t],sys.P["x",t])-np.dot(sys.R[t],sys.Q["x",t])
        Gz2=np.linalg.multi_dot([sys.R[t],Phi[t],sys.Xi["w",t]])+\
            np.linalg.multi_dot([sys.S[t],Theta[t],sys.Xi["w",t]])+\
            np.hstack(( np.dot(sys.D[t],sys.P["w",t]),np.zeros((sys.z,sys.n)) ))-\
            np.dot(sys.R[t],sys.Q["w",t])
        Gz3=np.linalg.multi_dot([sys.R[t],Phi[t],sys.Xi["v",t]])+\
            np.linalg.multi_dot([sys.S[t],Theta[t],sys.Xi["v",t]])+\
            -np.dot(sys.R[t],sys.Q["v",t])
        Gw=spa.block_diag(*[sys.W[t].G for tau in range(0,t+1)])
        Gv=spa.block_diag(*[sys.V[t].G for tau in range(0,t+1)])
        Gz=np.hstack(( np.dot(Gz1,sys.X0.G), np.dot(Gz2,Gw), np.dot(Gz3,Gv) ))
        Z[t]=zonotope(z_bar,Gz)
    for t in range(T):
        # U zonotope
        u_bar=u_tilde[t]+np.dot(theta[t],sys.Xi[t].x)
        Gu=np.dot(theta[t],sys.Xi[t].G)
        U[t]=zonotope(u_bar,Gu)
    # Proxy Quadratic cost
    for t in range(1,T):
        print t,"cost"
        prog.AddQuadraticCost(sum(U[t].x.flatten()**2))
        prog.AddQuadraticCost(sum(U[t].G.flatten()**2))
        prog.AddQuadraticCost(sum(Z[t].x.flatten()**2))
        prog.AddQuadraticCost(sum(Z[t].G.flatten()**2))
    print "Now solving the QP"
    result=gurobi_solver.Solve(prog,None,None)
    if result.is_success():
        print "Synthesis Success!","\n"*5
#        print "D=",result.GetSolution(D)
        theta_n={t:result.GetSolution(theta[t]).reshape(theta[t].shape) for t in range(0,T)}
        u_tilde_n={t:result.GetSolution(u_tilde[t]).reshape(u_tilde[t].shape) for t in range(0,T)}
        return u_tilde_n,theta_n
    else:
        print "Synthesis Failed!"            
        
def output_feedback_synthesis_lightweight_many_variables(sys,T,H_Z={},H_U={}):
    prog=MP.MathematicalProgram()
    # Add Variables
    phi,theta,Phi,Theta={},{},{},{}
    y_tilde,u_tilde={},{} 
    Y_tilde,U_tilde={},{}
    z_bar,u_bar={},{}
    Z,U={},{} # Z matrices
    Gz={}
    # Main variables
    for t in range(T+1):
        theta[t]=prog.NewContinuousVariables(sys.m,sys.o*(t+1),"theta%d"%t)
        u_tilde[t]=prog.NewContinuousVariables(sys.m,1,"u_tilde%d"%t)
        y_tilde[t]=prog.NewContinuousVariables(sys.o,1,"y_tilde%d"%t)
        phi[t]=prog.NewContinuousVariables(sys.o,sys.o*(t+1),"Phi%d"%t)
        Gz["var",t]=prog.NewContinuousVariables(sys.z,sys.Xi_reduced[t].G.shape[1],"Gz%d"%t)
    Theta[0]=theta[0]
    y_tilde[0]=np.zeros((sys.o,1))
    phi[0]=np.eye(sys.o)
    Phi[0]=phi[0]
    # Aggragates
    for t in range(T+1):
        Y_tilde[t]=np.vstack([y_tilde[tau] for tau in range(t+1)])
        U_tilde[t]=np.vstack([u_tilde[tau] for tau in range(t+1)])
    for t in range(T):
        # Dynamics
        s=np.dot(sys.M[t],Y_tilde[t])+np.dot(sys.N[t],U_tilde[t])
        prog.AddLinearConstraint(np.equal(y_tilde[t+1],s,dtype='object').flatten())
        S=np.hstack((np.dot(sys.M[t],Phi[t])+np.dot(sys.N[t],Theta[t]),np.eye(sys.o)))
        prog.AddLinearConstraint(np.equal(phi[t+1],S,dtype='object').flatten())
        Phi[t+1]=triangular_stack(Phi[t],phi[t+1])
        Theta[t+1]=triangular_stack(Theta[t],theta[t+1])
        # Performance variables
    for t in range(T+1):
        # Z zonotope
        print "time construction",t
        z_bar=np.linalg.multi_dot([sys.R[t],Y_tilde[t]])+\
            np.linalg.multi_dot([sys.R[t],Phi[t],sys.Xi[t].x])+\
            np.linalg.multi_dot([sys.S[t],U_tilde[t]])+\
            np.linalg.multi_dot([sys.S[t],Theta[t],sys.Xi_reduced[t].x])+\
            sys.F[t].x
        Gz[t]=np.linalg.multi_dot([sys.R[t],Phi[t],sys.Xi_reduced[t].G])+\
            np.linalg.multi_dot([sys.S[t],Theta[t],sys.Xi_reduced[t].G])
        Z[t]=zonotope(z_bar,np.hstack((Gz[t],sys.F_reduced[t].G)))
#        Z[t]=zonotope(z_bar,Gz[t])
        prog.AddLinearConstraint(np.equal(Gz[t],Gz["var",t],dtype='object').flatten())
    for t in range(T):
        # U zonotope
        u_bar=u_tilde[t]+np.dot(theta[t],sys.Xi_reduced[t].x)
        Gu=np.dot(theta[t],sys.Xi_reduced[t].G)
        U[t]=zonotope(u_bar,Gu)
    # Constraints
    for t,value in H_U.items():
        print "performing subset for U",t,value
        subset(prog,U[t],H_U[t]) 
    for t,value in H_Z.items():
        print "performing subset for Z",t,value
        subset(prog,Z[t],H_Z[t]) 
    # Proxy Linear Cost
    if True:
        r={}
        r["u-max"]=prog.NewContinuousVariables(1,1,"ru-max")
        r["z-max"]=prog.NewContinuousVariables(1,1,"rz-max")
        prog.NewContinuousVariables(1,1,"r")
#        prog.AddLinearCost(r["u-max"][0,0])
#        prog.AddLinearCost(r["z-max"][0,0])
        for t in range(1,T+1):
            print t,"adding cost for z"
            r["z",t]=prog.NewContinuousVariables(1,1,"r")
            R=hyperbox(sys.z)
            ZT=H_polytope(R.H_polytope.H,np.dot(R.H_polytope.h,r["z",t]),symbolic=True)
            subset(prog,Z[t],ZT) 
#            prog.AddLinearConstraint(np.less_equal(r["z",t],r["z-max"],dtype='object').flatten())
            prog.AddQuadraticCost(r["z",t][0,0]*r["z",t][0,0])
        for t in range(T):
            print t,"adding cost for u"
            r["u",t]=prog.NewContinuousVariables(1,1,"r")
            R=hyperbox(sys.m)
            UT=H_polytope(R.H_polytope.H,np.dot(R.H_polytope.h,r["u",t]),symbolic=True)
            subset(prog,U[t],UT) 
            prog.AddQuadraticCost(r["u",t][0,0]*r["u",t][0,0])
#            subset(prog,U[t],H_polytope(R.H_polytope.H,np.dot(R.H_polytope.h,3)))
#            prog.AddLinearConstraint(np.less_equal(r["u",t],r["u-max"],dtype='object').flatten())

#     Proxy Quadratic cost
    elif False:
        J=0
        for t in range(T):
            print t,"cost"
            J+=sum(U[t].x.flatten()**2)
            J+=sum(U[t].G.flatten()**2)
            J+=sum(Z[t+1].x.flatten()**2)
            J+=sum(Gz[t+1].flatten()**2)
        prog.AddQuadraticCost(J)    
    print "Now solving the Linear Program"
    start=time.time()
    result=gurobi_solver.Solve(prog,None,None)
    print "time to solve",time.time()-start
    if result.is_success():
        print "Synthesis Success!","\n"*5
#        print "D=",result.GetSolution(D)
        theta_n={t:result.GetSolution(theta[t]).reshape(theta[t].shape) for t in range(0,T)}
        u_tilde_n={t:result.GetSolution(u_tilde[t]).reshape(u_tilde[t].shape) for t in range(0,T)}
        return u_tilde_n,theta_n
    else:
        print "Synthesis Failed!"

        
def outputfeedback_synthesis_zonotope_solution(sys,u_tilde,theta):
    T=max(theta.keys())+1
    phi,Phi,Theta={},{},{}
    y_tilde={}
    Y_tilde,U_tilde={},{}
    z_bar,u_bar={},{}
    Z,U={},{} # Z matrices
    # Initial
    y_tilde[0]=np.zeros((sys.o,1))
    phi[0]=np.eye(sys.o)
    # Yashasin
    Phi[0],Theta[0]=phi[0],theta[0]
    theta[T]=np.zeros((sys.m,sys.o*(T+1)))
    u_tilde[T]=np.zeros((sys.m,1))
    for t in range(T):
        Y_tilde[t]=np.vstack([y_tilde[tau] for tau in range(t+1)])
        U_tilde[t]=np.vstack([u_tilde[tau] for tau in range(t+1)])
        y_tilde[t+1]=np.dot(sys.M[t],Y_tilde[t])+np.dot(sys.N[t],U_tilde[t])
        phi[t+1]=np.hstack(( np.dot(sys.M[t],Phi[t])+np.dot(sys.N[t],Theta[t]),np.eye(sys.o) ))
        Phi[t+1]=triangular_stack(Phi[t],phi[t+1])
        Theta[t+1]=triangular_stack(Theta[t],theta[t+1])
    Y_tilde[T]=np.vstack([y_tilde[tau] for tau in range(T+1)])
    U_tilde[T]=np.vstack([u_tilde[tau] for tau in range(T+1)])
    # Performance variables
    for t in range(T+1):
        # Z zonotope
        print "time synthesis",t
        z_bar=np.linalg.multi_dot([sys.R[t],Y_tilde[t]])+\
            np.linalg.multi_dot([sys.R[t],Phi[t],sys.Xi[t].x])+\
            np.linalg.multi_dot([sys.S[t],U_tilde[t]])+\
            np.linalg.multi_dot([sys.S[t],Theta[t],sys.Xi[t].x])+\
            sys.F[t].x
        Gz1=np.linalg.multi_dot([sys.R[t],Phi[t],sys.Xi["x",t]])+\
            np.linalg.multi_dot([sys.S[t],Theta[t],sys.Xi["x",t]])+\
            np.dot(sys.D[t],sys.P["x",t])-np.dot(sys.R[t],sys.Q["x",t])
        Gz2=np.linalg.multi_dot([sys.R[t],Phi[t],sys.Xi["w",t]])+\
            np.linalg.multi_dot([sys.S[t],Theta[t],sys.Xi["w",t]])+\
            np.hstack(( np.dot(sys.D[t],sys.P["w",t]),np.zeros((sys.z,sys.n)) ))-\
            np.dot(sys.R[t],sys.Q["w",t])
        Gz3=np.linalg.multi_dot([sys.R[t],Phi[t],sys.Xi["v",t]])+\
            np.linalg.multi_dot([sys.S[t],Theta[t],sys.Xi["v",t]])+\
            -np.dot(sys.R[t],sys.Q["v",t])
        Gw=spa.block_diag(*[sys.W[t].G for tau in range(0,t+1)])
        Gv=spa.block_diag(*[sys.V[t].G for tau in range(0,t+1)])
        Gz=np.hstack(( np.dot(Gz1,sys.X0.G), np.dot(Gz2,Gw), np.dot(Gz3,Gv) ))
        Z[t]=zonotope(z_bar,Gz)
    for t in range(T):
        # U zonotope
        u_bar=u_tilde[t]+np.dot(theta[t],sys.Xi[t].x)
        Gu=np.dot(theta[t],sys.Xi[t].G)
        U[t]=zonotope(u_bar,Gu)
    return Z,U

def triangular_stack(A,B):
    q=B.shape[1]-A.shape[1]
    if q>=0:
        return np.vstack((np.hstack((A,np.zeros((A.shape[0],q)))),B))
    else:
        return np.vstack((A,np.hstack((B,np.zeros((B.shape[0],-q))))))

def triangular_stack_list(list_of_matrices):
    N=len(list_of_matrices)
    if N==0:
        raise NotImplementedError
    elif N==1:
        return list_of_matrices[0]
    else:
        J=triangular_stack(list_of_matrices[0],list_of_matrices[1])
        for t in range(2,N):
            J=triangular_stack(J,list_of_matrices[t])
        return J
#def theta_encoding(sys,prog,T):
    # Add Variables        
    