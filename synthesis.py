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

def output_feedback_synthesis_lightweight(sys,T):
    prog=MP.MathematicalProgram()
    # Add Variables
    phi,theta,Phi,Theta={},{},{},{}
    y_tilde,u_tilde={},{} 
    Y_tilde,U_tilde={},{}
    z_bar,u_bar={},{}
    Z,U={},{} # Z matrices
    Gz={}
    # Initial Condition
    y_tilde[0]=np.zeros((sys.o,1))
    phi[0]=np.eye(sys.o)
    # Main variables
    for t in range(T):
        theta[t]=prog.NewContinuousVariables(sys.m,sys.o*(t+1),"theta%d"%t)
        u_tilde[t]=prog.NewContinuousVariables(sys.m,1,"u_tilde%d"%t)
    for t in range(T+1):
        Gz["var",t]=prog.NewContinuousVariables(sys.z,sys.Xi_reduced[t].G.shape[1],"Gz%d"%t)
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
            np.linalg.multi_dot([sys.S[t],Theta[t],sys.Xi_reduced[t].x])+\
            sys.F[t].x
        Gz[t]=np.linalg.multi_dot([sys.R[t],Phi[t],sys.Xi_reduced[t].G])+\
            np.linalg.multi_dot([sys.S[t],Theta[t],sys.Xi_reduced[t].G])
        Z[t]=zonotope(z_bar,np.hstack((Gz[t],sys.F_reduced[t].G)))
        prog.AddLinearConstraint(np.equal(Gz[t],Gz["var",t],dtype='object').flatten())
    for t in range(T):
        # U zonotope
        u_bar=u_tilde[t]+np.dot(theta[t],sys.Xi_reduced[t].x)
        Gu=np.dot(theta[t],sys.Xi_reduced[t].G)
        U[t]=zonotope(u_bar,Gu)
    # Proxy Linear Cost
#    r=prog.NewContinuousVariables(1,1,"r")
#    R=hyperbox(sys.z)
#    ZT=H_polytope(R.H_polytope.H,np.dot(R.H_polytope.h,r),symbolic=True)
#    print Z[T].G.shape,ZT
#    subset(prog,Z[T],ZT)    
    # Proxy Quadratic cost
#    J=0
#    for t in range(1,T):
#        print t,"cost"
#        J+=sum(U[t].x.flatten()**2)
#        J+=sum(U[t].G.flatten()**2)
#        J+=sum(Z[t].x.flatten()**2)
#        J+=sum(Gz[t].flatten()**2)
    print "Now adding the hugggggee Cost function"
#    print Phi[5]
#    print theta_all.shape
#    a=J.Jacobian(theta_all)
#    print a.shape
#    raise 1
#    return theta,J
#    return J.Jacobian(theta[T-1].T)
    for t in range(T):
        prog.AddQuadraticCost(np.eye(theta[t].shape[1]),np.zeros(theta[t].shape[1]),theta[t].T)
        prog.AddQuadraticCost(sum(Gz["var",t].flatten()**2))
#    prog.AddLinearCost(1000*r[0,0])
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
        
def output_feedback_synthesis_lightweight_many_variables(sys,T):
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
        prog.AddLinearConstraint(np.equal(Gz[t],Gz["var",t],dtype='object').flatten())
    for t in range(T):
        # U zonotope
        u_bar=u_tilde[t]+np.dot(theta[t],sys.Xi_reduced[t].x)
        Gu=np.dot(theta[t],sys.Xi_reduced[t].G)
        U[t]=zonotope(u_bar,Gu)
    # Proxy Linear Cost
    r=prog.NewContinuousVariables(1,1,"r")
    R=hyperbox(sys.z)
    ZT=H_polytope(R.H_polytope.H,np.dot(R.H_polytope.h,r),symbolic=True)
#    print Z[T].G.shape,ZT
    subset(prog,Z[T],ZT)    
#     Proxy Quadratic cost
#    J=0
#    for t in range(1,T):
#        print t,"cost"
#        J+=sum(U[t].x.flatten()**2)
#        J+=sum(U[t].G.flatten()**2)
#        J+=sum(Z[t].x.flatten()**2)
#        J+=sum(Gz[t].flatten()**2)
    print "Now adding the hugggggee Cost function"
    for t in range(T):
        prog.AddQuadraticCost(np.eye(theta[t].shape[1]),np.zeros(theta[t].shape[1]),theta[t].T)
#        prog.AddQuadraticCost(sum(Gz["var",t].flatten()**2))
    prog.AddLinearCost(100*r[0,0])
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

def synthesis(sys,T,y_goal,q0=1,control_bound=False):
    prog=MP.MathematicalProgram()
    phi,theta,Phi,Theta={},{},{},{}
    ybar,ubar={},{}
    phi[0],ybar[0]=Girard(np.dot(sys.C[0],sys.X0.G),q0),np.dot(sys.C[0],sys.X0.x)+sys.V[0].x
    theta[0]=prog.NewContinuousVariables(sys.m,phi[0].shape[1],"theta_0")
    Phi[0],Theta[0]=phi[0],theta[0]
    for t in range(T+1):
        ybar[t+1]=prog.NewContinuousVariables(sys.o,1,"ybar%d"%t)
        ubar[t]=prog.NewContinuousVariables(sys.m,1,"ubar%d"%t)
#        print "variables",t,Phi[t].shape,Theta[t].shape,sys.Z[t].G.shape
        phi[t+1]=prog.NewContinuousVariables(sys.o,phi[t].shape[1]+sys.Z[t].G.shape[1],"phi%d"%t)
        theta[t+1]=prog.NewContinuousVariables(sys.m,theta[t].shape[1]+sys.Z[t].G.shape[1],"phi%d"%t)
#        print "phi[t+1]",phi[t+1].shape,"theta[t+1]",theta[t+1].shape
        Phi[t+1]=np.vstack(( np.hstack((Phi[t],np.zeros((Phi[t].shape[0],sys.Z[t].G.shape[1])) )),\
                           phi[t+1] ))
        Theta[t+1]=np.vstack(( np.hstack((Theta[t],np.zeros((Theta[t].shape[0],sys.Z[t].G.shape[1])) )),\
                           theta[t+1] ))
    # Dynamic Constraints
    for t in range(T):
        print "Dynamics",t
        Ybar=np.vstack([ybar[tau] for tau in range(t+1)])
        Ubar=np.vstack([ubar[tau] for tau in range(t+1)])
#        print "Ybar:",Ybar.shape,"Ubar:",Ybar.shape
        s=np.dot(sys.M[t],Ybar)+np.dot(sys.N[t],Ubar)+sys.Z[t].x
        prog.AddLinearConstraint(np.equal(ybar[t+1],s,dtype='object'))
        S=np.hstack((np.dot(sys.M[t],Phi[t])+np.dot(sys.N[t],Theta[t]),sys.Z[t].G))
        prog.AddLinearConstraint(np.equal(phi[t+1],S,dtype='object').flatten())
    # Final Constarint
    Z_f=zonotope(ybar[T],phi[T])
    # Cost!
    for t in range(T):
        prog.AddQuadraticCost(50*np.eye(sys.m),np.zeros(sys.m),ybar[t+1])
        prog.AddQuadraticCost(5*np.trace(np.dot(phi[t].T,phi[t])))
        prog.AddQuadraticCost(1*np.eye(sys.m),np.zeros(sys.m),ubar[t])
        prog.AddQuadraticCost(0.1*np.trace(np.dot(theta[t].T,theta[t])))
    prog.AddQuadraticCost(100*np.trace(np.dot(phi[T].T,phi[T])))
    # Control subset
#    for t in range(T):
#        Z_theta=zonotope(ubar[t],theta[t])
#        subset(prog,Z_theta,sys.U_set)
    # Final Subset
#    subset(prog,Z_f,y_goal)
#    D=subset_soft(prog,Z_f,y_goal)
    result=gurobi_solver.Solve(prog,None,None)
    print result
    if result.is_success():
        print "Synthesis Success!","\n"*5
#        print "D=",result.GetSolution(D)
        sys.Phi={t:sym.Evaluate(result.GetSolution(Phi[t]),{}) for t in range(1,T+1)}
        sys.Theta={t:sym.Evaluate(result.GetSolution(Theta[t]),{}) for t in range(T)}
        sys.ybar={t:result.GetSolution(ybar[t]) for t in range(1,T+1)}
        sys.ubar={t:result.GetSolution(ubar[t]) for t in range(T)}
        sys.phi={t:result.GetSolution(phi[t]) for t in range(1,T+1)}
        sys.theta={t:result.GetSolution(theta[t]) for t in range(T)}
        sys.phi[0],sys.Phi[0]=phi[0],phi[0]
        sys.ybar[0]=ybar[0]
    else:
        print "Synthesis Failed!"
        
def synthesis_disturbance_feedback(sys,T,y_goal,control_bound=False):
    prog=MP.MathematicalProgram()
    phi,theta,Phi,Theta={},{},{},{}
    ybar,ubar={},{}
    phi[0]=Girard(np.dot(sys.C[0],sys.X0.G),sys.E[-1].G.shape[0])
    theta[0]=prog.NewContinuousVariables(sys.m,phi[0].shape[1],"theta_0")
    ybar[0]=np.dot(sys.C[0],sys.X0.x)+sys.V[0].x
    Phi[0],Theta[0]=phi[0],theta[0]
    phi_E,theta_E={},{}
    for t in range(T+1):
        ybar[t+1]=prog.NewContinuousVariables(sys.o,1,"ybar%d"%t)
        ubar[t]=prog.NewContinuousVariables(sys.m,1,"ubar%d"%t)
        phi[t+1]=prog.NewContinuousVariables(sys.o,sys.E[t].x.shape[0],"phi%d"%t)
        theta[t+1]=prog.NewContinuousVariables(sys.m,sys.E[t].x.shape[0],"phi%d"%t)
        Phi[t+1]=np.vstack(( np.hstack((Phi[t],np.zeros((Phi[t].shape[0],sys.E[t].x.shape[0]-sys.E[t-1].x.shape[0])) )),\
                           phi[t+1] ))
        Theta[t+1]=np.vstack(( np.hstack((Theta[t],np.zeros((Theta[t].shape[0],sys.E[t].x.shape[0]-sys.E[t-1].x.shape[0])) )),\
                           theta[t+1] ))
    # Dynamic Constraints
    print "Adding Dynamical Constraints"
    for t in range(T):
        print t,
        Ybar=np.vstack([ybar[tau] for tau in range(t+1)])
        Ubar=np.vstack([ubar[tau] for tau in range(t+1)])
        s=np.dot(sys.M[t],Ybar)+np.dot(sys.N[t],Ubar)+sys.ebar[t]
        prog.AddLinearConstraint(np.equal(ybar[t+1],s,dtype='object'))
        S=np.hstack((np.dot(sys.M[t],Phi[t])+np.dot(sys.N[t],Theta[t]),np.eye(sys.o)))
        prog.AddLinearConstraint(np.equal(phi[t+1],S,dtype='object').flatten())
    # The cosmetic variables: zonotope generators
    for t in range(0,T+1):
        phi_E[t]=np.dot(phi[t],sys.E[t-1].G)
        theta_E[t]=np.dot(theta[t],sys.E[t-1].G)
    # Final Constarint
    Z_f=zonotope(ybar[T],phi_E[T])
    # Cost!
    for t in range(T):
        prog.AddQuadraticCost(1*np.eye(sys.m),np.zeros(sys.m),ybar[t+1])
        prog.AddQuadraticCost(1*np.trace(np.dot(phi_E[t].T,phi_E[t])))
        prog.AddQuadraticCost(1*np.eye(sys.m),np.zeros(sys.m),ubar[t])
        prog.AddQuadraticCost(1*np.trace(np.dot(theta_E[t].T,theta_E[t])))
    # Terminal Cost
    prog.AddQuadraticCost(0.01*np.trace(np.dot(phi_E[T].T,phi_E[T])))
    # Control subset
    if control_bound:
        for t in range(T):
            Z_theta=zonotope(ubar[t],theta_E[t])
            subset(prog,Z_theta,sys.U_set)
    # Final Subset
#    subset(prog,Z_f,y_goal)
#    D=subset_soft(prog,Z_f,y_goal)
    result=gurobi_solver.Solve(prog,None,None)
    print result
    if result.is_success():
        print "\n"*5, "*"*10,"Synthesis Success!","*"*10,"\n"*5
#        print "D=",result.GetSolution(D)
        sys.Phi={t:sym.Evaluate(result.GetSolution(Phi[t]),{}) for t in range(1,T+1)}
        sys.Theta={t:sym.Evaluate(result.GetSolution(Theta[t]),{}) for t in range(T)}
        sys.ybar={t:result.GetSolution(ybar[t]) for t in range(1,T+1)}
        sys.ubar={t:result.GetSolution(ubar[t]) for t in range(T)}
        sys.phi={t:result.GetSolution(phi[t]) for t in range(1,T+1)}
        sys.theta={t:result.GetSolution(theta[t]) for t in range(T)}
        sys.phi[0],sys.Phi[0]=phi[0],phi[0]
        sys.ybar[0]=ybar[0]
        sys.phi_E={t:np.dot(sys.phi[t],sys.E[t-1].G) for t in range(T+1)}
        sys.theta_E={t:np.dot(sys.theta[t],sys.E[t-1].G) for t in range(T)}
    else:
        print "Synthesis Failed!"            

def zonotopic_controller(x_current,X,U):
    q=X.G.shape[1]
    prog=MP.MathematicalProgram()
    zeta=prog.NewContinuousVariables(q,1,"zeta")
    epsilon=prog.NewContinuousVariables(1,1,"epsilon")
#    prog.AddBoundingBoxConstraint(-1,1,zeta) # zeta
    prog.AddLinearEqualityConstraint(X.G,x_current-X.x,zeta)
    prog.AddLinearConstraint(A=np.hstack((np.eye(q),-np.ones((q,1)))),\
                             lb=-np.inf*np.ones((q,1)),ub=np.zeros((q,1)),vars=np.vstack((zeta,epsilon)))
    prog.AddLinearEqualityConstraint(X.G,x_current-X.x,zeta)
    prog.AddLinearConstraint(A=np.hstack((np.eye(q),np.ones((q,1)))),\
                             lb=np.zeros((q,1)),ub=np.inf*np.ones((q,1)),vars=np.vstack((zeta,epsilon)))
    prog.AddLinearCost(epsilon[0,0])
    result=gurobi_solver.Solve(prog,None,None)
    if result.is_success():
        print "controller solution found! with",result.GetSolution(epsilon)
        zeta_num=result.GetSolution(zeta).reshape(zeta.shape[0],1)  
        zeta_num=project_to_box(zeta_num)
#        print "zeta is",zeta_num.T,zeta_num.shape
        return (np.dot(U.G,zeta_num)+U.x).reshape(U.x.shape[0],1)
    else:
        print result
        print "controller solution not found!"
        print "x_current:",x_current.T-X.x.T
        print "X:",X.G
        
def zonotopic_controller_soft(x_current,X,U):
    """
    x+delta=bar{x}+Gzeta
    """
    q=X.G.shape[1]
    n=X.G.shape[0]
    prog=MP.MathematicalProgram()
    zeta=prog.NewContinuousVariables(q,1,"zeta")
    delta=prog.NewContinuousVariables(n,1,"delta")
    epsilon=prog.NewContinuousVariables(1,1,"epsilon")
    prog.AddBoundingBoxConstraint(-1,1,zeta) # zeta
    prog.AddLinearEqualityConstraint(np.hstack((X.G,-np.eye(n))),x_current-X.x,np.vstack((zeta,delta)))
    prog.AddLinearConstraint(A=np.hstack((np.eye(q),-np.ones((q,1)))),\
                             lb=-np.inf*np.ones((q,1)),ub=np.zeros((q,1)),vars=np.vstack((zeta,epsilon)))
    prog.AddLinearEqualityConstraint(X.G,x_current-X.x,zeta)
    prog.AddLinearConstraint(A=np.hstack((np.eye(q),np.ones((q,1)))),\
                             lb=np.zeros((q,1)),ub=np.inf*np.ones((q,1)),vars=np.vstack((zeta,epsilon)))
    prog.AddLinearCost(epsilon[0,0])
    prog.AddQuadraticCost(1*np.eye(n),np.zeros(n),delta)
    result=gurobi_solver.Solve(prog,None,None)
    if result.is_success():
        print "controller solution found! with",result.GetSolution(epsilon),result.GetSolution(delta)
        zeta_num=result.GetSolution(zeta).reshape(zeta.shape[0],1)  
#        print "zeta is",zeta_num.T,zeta_num.shape
        return (np.dot(U.G,zeta_num)+U.x).reshape(U.x.shape[0],1)
    else:
        print "controller solution not found!"
        print "x_current:",x_current
        

def to_zonotope_control(A,B,c,x_current,X_next,U):
    raise NotImplementedError
    q=X_next.G.shape[1]
    prog=MP.MathematicalProgram()
    zeta=prog.NewContinuousVariables(q,1,"zeta")
    epsilon=prog.NewContinuousVariables(1,1,"epsilon")
    x_next=prog.NewContinuousVariables(x_current.shape[0],1,"x_next")
#    prog.AddBoundingBoxConstraint(-100,100,zeta) # zeta
    prog.AddLinearEqualityConstraint(X_next.G,x_current-X.x,zeta)
    prog.AddLinearConstraint(A=np.hstack((np.eye(q),-np.ones((q,1)))),\
                             ub=np.zeros((q,1)),lb=-np.inf*np.ones((q,1)),vars=np.vstack((zeta,epsilon)))
    prog.AddLinearEqualityConstraint(X.G,x_current-X.x,zeta)
    prog.AddLinearConstraint(A=np.hstack((np.eye(q),np.ones((q,1)))),\
                             lb=np.zeros((q,1)),ub=np.inf*np.ones((q,1)),vars=np.vstack((zeta,epsilon)))
    prog.AddLinearCost(epsilon[0,0])
    result=gurobi_solver.Solve(prog,None,None)
    if result.is_success():
        print "controller solution found! with",result.GetSolution(epsilon)
#        zeta_num=result.GetSolution(zeta).reshape(zeta.shape[0],1)  
#        print "zeta is",zeta_num.T,zeta_num.shape
        return np.dot(U.G,zeta_num)+U.x
    else:
        print "controller solution not found!"
        print "X:",X.x,X.G
        print "x_current:",x_current
        
def project_to_box(zeta):
    """
    Project on the unit cube
    """
    a=np.ones(zeta.shape)
    _z=np.minimum(a,zeta)
    return np.maximum(_z,-a)

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
    