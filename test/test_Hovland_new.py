#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 19:06:49 2019

@author: sadra
"""
import numpy as np
from pypolytrajectory.LTV import system,test_controllability
from pypolytrajectory.reduction import reduced_order,order_reduction_error,error_construction,error_construction_old
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.zonotope_order_reduction.methods import G_cut,Girard
from pypolytrajectory.synthesis import synthesis_disturbance_feedback,zonotopic_controller,synthesis

import pydrake.systems.controllers as PC


S=system()
n=6
m=1
o=1
T=48
#np.random.seed(1)
S.X0=zonotope(np.array(([0,0,0,0,0,0])).reshape(6,1),np.eye(n)*1)
for t in range(T):
    S.A[t]=np.array([[0.28,0.25,-0.19,-0.22,0.03,-0.50],
                     [0.25,-0.47,0.30,0.17,-0.11,-0.11],
                     [-0.19,0.30,0.46,0.09,-0.02,-0.08],
                     [-0.22,0.17,0.09,0.60,-0.06,0.14],
                     [0.03,-0.11,-0.02,-0.06,0.46,-0.13],
                     [-0.50,-0.11,-0.08,0.14,-0.13,-0.23]]).reshape(6,6)
    S.B[t]=np.array([[1.0159,0,0.5988,1.8641,0,-1.2155]]).reshape(6,1)
    #S.C[t]=np.array([[1.292,0,0,0.2361,0.8428,0]]).reshape(1,6)
    S.C[t]=np.array([[1.29,0.24,0,0,0,0]]).reshape(1,6)
    S.W[t]=zonotope(np.zeros((n,1)),np.eye(n)*0.05)
    S.V[t]=zonotope(np.zeros((o,1)),np.eye(o)*0.05)

S.U_set=zonotope(np.zeros((m,1)),np.eye(m)*1)
S.construct_dimensions()
S.construct_E()
M,N,Z=reduced_order(S,T-1)

S.Z=Z
S.M=M
S.N=N

test_controllability(S.A[0],S.B[0])
test_controllability(S.A[0].T,S.C[0].T)
S.K,J_c=PC.DiscreteTimeLinearQuadraticRegulator(S.A[0],S.B[0],100*np.eye(n),np.eye(m))
S.L,J_o=PC.DiscreteTimeLinearQuadraticRegulator(S.A[0].T,S.C[0].T,100*np.eye(n),np.eye(o))


import matplotlib.pyplot as plt0
plt0.plot(range(T-1),[np.linalg.norm(Z[t].G[0,:],1) for t in range(T-1)],LineWidth=2,color='green')
plt0.plot(range(T-1),[np.linalg.norm(Z[t].G[0,:],1) for t in range(T-1)],'o',MarkerSize=3,color='black')

# Synthesis
Goal=zonotope(np.ones((1,1))*0,np.eye(1)*1)

T=45
if False:
    for t in range(T+2):
        print t,"-zonotope reduction"
        G=Girard(Z[t].G,1)
        S.Z[t].G=G
        print G
    synthesis(S,T=T,y_goal=Goal)
else:
    error_construction(S,T+1,q0=1)
    order_reduction_error(S,T+1,q=2)
    synthesis_disturbance_feedback(S,T=T,y_goal=Goal,control_bound=True)

    
#raise 1
def simulate_observer(sys,x_0,T):
    x,y,u,v,w,x_observer={},{},{},{},{},{}
    x[0]=x_0
    zeta_w,zeta_v=2*(np.random.random((sys.n,1))-0.5),2*(np.random.random((sys.o,1))-0.5)
    zeta_v=np.ones((sys.o,1))*(-1)**np.random.randint(1,3)
    v[0]=np.dot(sys.V[0].G,zeta_v)+sys.V[0].x
    y[0]=np.dot(sys.C[0],x[0])+v[0]    
    x_observer[0]=np.dot(np.linalg.pinv(S.C[0]),y[0])
    for t in range(T+1):
        print "simulating observer time:",t
        zeta_w,zeta_v=2*(np.random.random((sys.n,1))-0.5),2*(np.random.random((sys.o,1))-0.5)
        zeta_w=np.ones((sys.n,1))*(-1)**np.random.randint(1,3)
        zeta_v=np.ones((sys.o,1))*(-1)**np.random.randint(1,3)
        v[t+1]=np.dot(sys.V[t+1].G,zeta_v)+sys.V[t+1].x
        w[t]=np.dot(sys.W[t].G,zeta_w)+sys.W[t].x
        if t==T:
            return x,y,u,v,w,x_observer
        u[t]=np.dot(-S.K,x_observer[t])
        x[t+1]=np.dot(sys.A[t],x[t])+np.dot(sys.B[t],u[t])+w[t]
        y[t+1]=np.dot(sys.C[t+1],x[t+1])+v[t+1]
        x_observer[t+1]=np.dot(sys.A[t],x_observer[t])+np.dot(sys.B[t],u[t])+\
            np.dot(sys.L.T,y[t+1]-np.dot(sys.C[t+1],np.dot(sys.A[t],x_observer[t])+np.dot(sys.B[t],u[t])))
        print "control",t,u[t],u[t].shape
    return x,y,u,v,w,x_observer

def simulate(sys,x_0,T):
    x,y,u,v,w={},{},{},{},{}
    x[0]=x_0
    for t in range(T+1):
        print "simulating time:",t
        zeta_w,zeta_v=2*(np.random.random((sys.n,1))-0.5),2*(np.random.random((sys.o,1))-0.5)
        zeta_w=np.ones((sys.n,1))*(-1)**np.random.randint(1,3)
        zeta_v=np.ones((sys.o,1))*(-1)**np.random.randint(1,3)
        v[t]=np.dot(sys.V[t].G,zeta_v)+sys.V[t].x
        w[t]=np.dot(sys.W[t].G,zeta_w)+sys.W[t].x
        y[t]=np.dot(sys.C[t],x[t])+v[t]
        if t==T:
            return x,y,u,v,w
        Y=np.vstack([y[tau] for tau in range(t+1)])
        Ybar=np.vstack([sys.ybar[tau] for tau in range(t+1)])
        zono_Y=zonotope(Ybar,np.dot(sys.Phi[t],sys.E[t-1].G))
        zono_U=zonotope(sys.ubar[t],np.dot(sys.theta[t],sys.E[t-1].G))
        u[t]=zonotopic_controller(Y,zono_Y,zono_U)
#        u[t]=np.zeros((sys.m,1))
        print "control",t,u[t],u[t].shape
        x[t+1]=np.dot(sys.A[t],x[t])+np.dot(sys.B[t],u[t])+w[t]  
    return x,y,u,v,w

import matplotlib.pyplot as plt
fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
y_minus,y_plus,u_minus,u_plus=[],[],[],[]
for t in range(T+1):
    y_minus.append(np.asscalar(S.ybar[t]-np.linalg.norm(S.phi_E[t],ord=np.inf)))
    y_plus.append(np.asscalar(S.ybar[t]+np.linalg.norm(S.phi_E[t],ord=np.inf)))
ax0.fill_between(range(T+1),y_minus,y_plus,alpha=0.5,color='red')
y_minus,y_plus,u_minus,u_plus=[],[],[],[]
for t in range(T):
    u_minus.append(np.asscalar(S.ubar[t]-np.linalg.norm(S.theta_E[t],ord=np.inf)))
    u_plus.append(np.asscalar(S.ubar[t]+np.linalg.norm(S.theta_E[t],ord=np.inf)))
ax1.fill_between(range(T),u_minus,u_plus,color='green',alpha=0.8)
ax0.set_xlabel(r'time')
ax0.set_ylabel(r'$y$')
ax0.set_title(r'Reachable Outputs Over Time')
ax1.set_xlabel(r'time')
ax1.set_ylabel(r'$u$')
ax1.set_title(r'Possible Control Inputs Over Time')
for i in range(5):
    zeta_x=2*(np.random.random((S.n,1))-0.5)
    x_0=np.dot(S.X0.G,zeta_x)+S.X0.x
    x,y,u,v,w=simulate(S,x_0,T)
    x_o,y_o,u_o,v_o,w_o,x_observer=simulate_observer(S,x_0,T)
    # X axis
    ax0.plot(range(T+1),[np.asscalar(y[t]) for t in range(T+1)],'.',color='red')
    ax0.plot(range(T+1),[np.asscalar(y[t]) for t in range(T+1)],'-',color='red')
    ax0.plot(range(T+1),[np.asscalar(y_o[t]) for t in range(T+1)],'.',color='blue')
    ax0.plot(range(T+1),[np.asscalar(y_o[t]) for t in range(T+1)],'-',color='blue')
    ax1.plot(range(T),[np.asscalar(u[t]) for t in range(T)],'.',color='orange')
    ax1.plot(range(T),[np.asscalar(u[t]) for t in range(T)],'-',color='orange')
    ax1.plot(range(T),[np.asscalar(u_o[t]) for t in range(T)],'.',color='cyan')
    ax1.plot(range(T),[np.asscalar(u_o[t]) for t in range(T)],'-',color='cyan')
    for i in range(n):
        ax2.plot(range(T),[x_o[t][i,0]-x_observer[t][i,0] for t in range(T)],'-',color='blue')
        ax2.plot(range(T),[x_o[t][i,0]-x_observer[t][i,0] for t in range(T)],'.',color='blue')