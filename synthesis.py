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
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.containment_pydrake import subset,subset_soft
# use Gurobi solver
global gurobi_solver,OSQP_solver
gurobi_solver=Gurobi_drake.GurobiSolver()
OSQP_solver=OSQP_drake.OsqpSolver()

def synthesis(sys,T,q0,y_goal):
    prog=MP.MathematicalProgram()
    phi,theta,Phi,Theta={},{},{},{}
    ybar,ubar={},{}
    phi[0]=prog.NewContinuousVariables(sys.o,q0,"phi_0")
    theta[0]=prog.NewContinuousVariables(sys.m,q0,"theta_0")
    Phi[0],Theta[0]=phi[0],theta[0]
    for t in range(T+1):
        ybar[t]=prog.NewContinuousVariables(sys.o,1,"ybar%d"%t)
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
    c=np.linalg.norm(sys.X0.G,ord=np.inf)
    prog.AddBoundingBoxConstraint(-c,c,phi[0]) # zeta
    prog.AddLinearCost(sum([phi[0][i,i]*1000 for i in range(sys.o)]))
    prog.AddQuadraticCost(np.eye(sys.m),np.zeros(sys.m),ubar[0])
    prog.AddQuadraticCost(np.eye(sys.o),0*np.ones(sys.o),ybar[0])
    for t in range(T):
        prog.AddQuadraticCost(3*np.trace(np.dot(phi[t].T,phi[t])))
    for t in range(T):
        prog.AddQuadraticCost(np.trace(np.dot(theta[t].T,theta[t])))
    prog.AddQuadraticCost(30*np.trace(np.dot(phi[T].T,phi[T])))
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
        sys.Phi={t:sym.Evaluate(result.GetSolution(Phi[t]),{}) for t in range(T+1)}
        sys.Theta={t:sym.Evaluate(result.GetSolution(Theta[t]),{}) for t in range(T)}
        sys.ybar={t:result.GetSolution(ybar[t]) for t in range(T+1)}
        sys.ubar={t:result.GetSolution(ubar[t]) for t in range(T)}
        sys.phi={t:result.GetSolution(phi[t]) for t in range(T+1)}
        sys.theta={t:result.GetSolution(theta[t]) for t in range(T)}
    else:
        print "Synthesis Failed!"
            

def zonotopic_controller(x_current,X,U):
    q=X.G.shape[1]
    prog=MP.MathematicalProgram()
    zeta=prog.NewContinuousVariables(q,1,"zeta")
    epsilon=prog.NewContinuousVariables(1,1,"epsilon")
#    prog.AddBoundingBoxConstraint(-100,100,zeta) # zeta
    prog.AddLinearEqualityConstraint(X.G,x_current-X.x,zeta)
    prog.AddLinearConstraint(A=np.hstack((np.eye(q),-np.ones((q,1)))),\
                             ub=np.zeros((q,1)),lb=-np.inf*np.ones((q,1)),vars=np.vstack((zeta,epsilon)))
    prog.AddLinearEqualityConstraint(X.G,x_current-X.x,zeta)
    prog.AddLinearConstraint(A=np.hstack((np.eye(q),np.ones((q,1)))),\
                             lb=np.zeros((q,1)),ub=np.inf*np.ones((q,1)),vars=np.vstack((zeta,epsilon)))
    prog.AddLinearCost(epsilon[0,0])
    result=gurobi_solver.Solve(prog,None,None)
    if result.is_success():
        print "controller solution found! with",result.GetSolution(epsilon)
        zeta_num=result.GetSolution(zeta).reshape(zeta.shape[0],1)  
#        print "zeta is",zeta_num.T,zeta_num.shape
        return np.dot(U.G,zeta_num)+U.x
    else:
        print "controller solution not found!"
        print "X:",X.x,X.G
        print "x_current:",x_current