#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:57:08 2019

@author: sadra
"""

import numpy as np
from pypolytrajectory.LTV import system
from pypolytrajectory.reduction import reduced_order
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.zonotope_order_reduction.methods import G_cut,Girard_hull
from pypolytrajectory.synthesis import synthesis,zonotopic_controller

S=system()
n=6
m=1
o=1
T=30
#np.random.seed(1)
S.X0=zonotope(np.array(([0,0,0,0,0,0])).reshape(6,1),np.eye(n)*5)
for t in range(T):
    S.A[t]=np.array([[0.28,0.25,-0.19,-0.22,0.03,-0.50],
                     [0.25,-0.47,0.30,0.17,-0.11,-0.11],
                     [-0.19,0.30,0.46,0.09,-0.02,-0.08],
                     [-0.22,0.17,0.09,0.60,-0.06,0.14],
                     [0.03,-0.11,-0.02,-0.06,0.46,-0.13],
                     [-0.50,-0.11,-0.08,0.14,-0.13,-0.23]]).reshape(6,6)
    S.B[t]=np.array([[1.0159,0,0.5988,1.8641,0,-1.2155]]).reshape(6,1)
    S.C[t]=np.array([[1.292,0,0,0.2361,0.8428,0]]).reshape(1,6)
    S.W[t]=zonotope(np.zeros((n,1)),np.eye(n)*0.05)
    S.V[t]=zonotope(np.zeros((o,1)),np.eye(o)*0.05)

S.U_set=zonotope(np.zeros((m,1)),np.eye(m)*1)
S.construct_dimensions()
S.construct_E()
M,N,Z=reduced_order(S,T-1)

import matplotlib.pyplot as plt
plt.plot(range(T-1),[np.linalg.norm(Z[t].G,1) for t in range(T-1)],LineWidth=5,color='green')
plt.plot(range(T-1),[np.linalg.norm(Z[t].G,1) for t in range(T-1)],'o',MarkerSize=10,color='black')
plt.xlabel(r'time')
plt.ylabel(r'$\|z_c[t]\|_1$')
plt.title(r'Error over time')
#raise 1

for t in range(T-1):
    print t,"-zonotope reduction"
#    D,G=G_cut(Z[t],4,'osqp')
    G=Girard_hull(Z[t],3)
    Z[t].G=G
    print G

S.Z=Z
S.M=M
S.N=N

# Synthesis
T=27
Goal=zonotope(np.ones((1,1))*0,np.eye(1)*1)
synthesis(S,q0=3,T=T,y_goal=Goal)

def simulate(sys,x_0,T):
    x,y,u,v,w={},{},{},{},{}
    x[0]=x_0
    for t in range(T+1):
        print "simulating time:",t
        zeta_w,zeta_v=2*(np.random.random((sys.n,1))-0.5),2*(np.random.random((sys.o,1))-0.5)
        zeta_w=np.random.randint(-1,1,size=(sys.n,1))
        zeta_v=np.random.randint(-1,1,size=(sys.m,1))
        v[t]=np.dot(sys.V[t].G,zeta_v)+sys.V[t].x
        w[t]=np.dot(sys.W[t].G,zeta_w)+sys.W[t].x
        y[t]=np.dot(sys.C[t],x[t])+v[t]
#        print "y",y[t].shape,y
        if t==0:
            Zono_Y,Zono_U=zonotope(sys.ybar[t],sys.Phi[0]),zonotope(sys.ubar[t],sys.Theta[0])
        elif t==T:
            print "End of simulation"
            return x,y,u,v,w
        else:
            print t,"continue",t==T,T
            YU=np.vstack([sys.ybar[tau] for tau in range(t+1)]+[sys.ubar[tau] for tau in range(t)])
            Zono_Y,Zono_U=zonotope(YU,np.vstack((\
                sys.Phi[t],np.hstack((sys.Theta[t-1],np.zeros((sys.Theta[t-1].shape[0],sys.Z[t].G.shape[1])))) ))),\
                                                 zonotope(sys.ubar[t],sys.Theta[t][sys.m*t:,:])
        H=np.vstack([y[tau] for tau in range(t+1)]+[u[tau] for tau in range(t)])
        u[t]=zonotopic_controller(H,Zono_Y,Zono_U)
#        print "control",t,u[t],u[t].shape
        x[t+1]=np.dot(sys.A[t],x[t])+np.dot(sys.B[t],u[t])+w[t]  
    return x,y,u,v,w

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
y_minus,y_plus,u_minus,u_plus=[],[],[],[]
for t in range(T+1):
    y_minus.append(np.asscalar(S.ybar[t]-np.linalg.norm(S.Phi[t][-1,:],ord=1)))
    y_plus.append(np.asscalar(S.ybar[t]+np.linalg.norm(S.Phi[t][-1,:],ord=1)))
ax.plot([0,T],[np.asscalar(Goal.x-Goal.G),np.asscalar(Goal.x-Goal.G)],'--',color='black')
ax.plot([0,T],[np.asscalar(Goal.x+Goal.G),np.asscalar(Goal.x+Goal.G)],'--',color='black')
ax.fill_between(range(T+1),y_minus,y_plus,alpha=0.3,color='red')
ax.set_xlabel('time')
ax.set_ylabel('y')
ax.set_title('Reachable Outputs Over Time')
for i in range(20):
    zeta_x=2*(np.random.random((S.n,1))-0.5)
    x_0=np.dot(S.X0.G,zeta_x)+S.X0.x
    x,y,u,v,w=simulate(S,x_0,T)
    # X axis
    plt.plot(range(T+1),[np.asscalar(y[t]) for t in range(T+1)],'o',color='red')
    plt.plot(range(T+1),[np.asscalar(y[t]) for t in range(T+1)],'-',color='red')
    for t in range(T):
        u_minus.append(np.asscalar(S.ubar[t]-np.linalg.norm(S.Theta[t][-1,:],ord=1)))
        u_plus.append(np.asscalar(S.ubar[t]+np.linalg.norm(S.Theta[t][-1,:],ord=1)))

    #plt.fill_between(range(T),u_minus,u_plus,color='red')
