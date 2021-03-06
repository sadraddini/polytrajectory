#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:19:27 2019

@author: sadra
"""

import numpy as np
from pypolytrajectory.LTV import system
from pypolytrajectory.reduction import reduced_order
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.zonotope_order_reduction.methods import G_cut,Girard_hull
from pypolytrajectory.synthesis import synthesis,zonotopic_controller

S=system()
n=4
m=1
o=1
T=55
np.random.seed(10)
S.X0=zonotope(np.ones((n,1))*0,np.eye(n)*1)
B=np.random.randint(0,2,size=(n,m))
A=np.eye(n)+np.random.normal(size=(n,n))*0.02
for t in range(T):
    S.A[t]=A
    S.B[t]=B
    S.C[t]=np.zeros((o,n))
    S.C[t][0,0]=1
    S.W[t]=zonotope(np.zeros((n,1)),np.eye(n)*0.01)
    S.V[t]=zonotope(np.zeros((o,1)),np.eye(o)*0.01)

S.U_set=zonotope(np.zeros((m,1)),np.eye(m)*2)
S.construct_dimensions()
S.construct_E()
M,N,Z=reduced_order(S,T-1)

#raise 1

for t in range(T-1):
    print t,"-zonotope reduction"
#    D,G=G_cut(Z[t],4,'osqp')
    G=Girard_hull(Z[t],1)
    Z[t].G=G
    print G

import matplotlib.pyplot as plt0
plt0.plot(range(T-1),[np.linalg.norm(Z[t].G,1) for t in range(T-1)],LineWidth=2,color='green')
plt0.plot(range(T-1),[np.linalg.norm(Z[t].G,1) for t in range(T-1)],'o',MarkerSize=3,color='black')

S.Z=Z
S.M=M
S.N=N

# Synthesis
T=40
Goal=zonotope(np.ones((1,1))*0,np.eye(1)*1)
synthesis(S,T=T,y_goal=Goal)


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
#        print "y",y[t].shape,y
        if t==0:
            Zono_Y,Zono_U=zonotope(sys.ybar[t],sys.phi[0]),zonotope(sys.ubar[t],sys.theta[0])
        elif t==T:
            print "End of simulation"
            return x,y,u,v,w
        else:
            print t,"continue",t==T,T
            YU=np.vstack([sys.ybar[tau] for tau in range(t+1)]+[sys.ubar[tau] for tau in range(t)])
            Zono_Y,Zono_U=zonotope(YU,np.vstack((\
                sys.Phi[t],np.hstack((sys.Theta[t-1],np.zeros((sys.Theta[t-1].shape[0],sys.Z[t].G.shape[1])))) ))),\
                                                 zonotope(sys.ubar[t],sys.theta[t])
        H=np.vstack([y[tau] for tau in range(t+1)]+[u[tau] for tau in range(t)])
#        try:
        Zono_Y=zonotope(np.vstack([sys.ybar[tau] for tau in range(t+1)]),sys.Phi[t])
        H=np.vstack([y[tau] for tau in range(t+1)])
        u[t]=zonotopic_controller(H,Zono_Y,Zono_U)
##        except:
##        if t>0:
##            H=np.vstack([y[tau] for tau in range(t+1)]+[u[tau] for tau in range(t)])
##            K=np.dot(sys.theta[t],np.linalg.pinv(np.vstack((sys.Phi[t],
##                     np.hstack((sys.Theta[t-1],np.zeros((sys.Theta[t-1].shape[0],sys.Z[t].G.shape[1]))))))))
##            u[t]=sys.ubar[t]+np.dot(K,H-YU)
##        elif t==0:
##            K=np.dot(sys.theta[0],np.linalg.pinv(sys.phi[0]))
##            u[t]=sys.ubar[t]+np.dot(K,y[0])
        u[t]=u[t].reshape(sys.m,1)
        print "control",t,u[t],u[t].shape
#        u[t]=np.zeros((sys.m,1))
        x[t+1]=np.dot(sys.A[t],x[t])+np.dot(sys.B[t],u[t])+w[t]  
    return x,y,u,v,w

def simulate_zero(sys,x_0,T):
    x,y,u,v,w={},{},{},{},{}
    x[0]=x_0
    for t in range(T+1):
        zeta_w,zeta_v=2*(np.random.random((sys.n,1))-0.5),2*(np.random.random((sys.o,1))-0.5)
        zeta_w=np.ones((sys.n,1))*(-1)**np.random.randint(1,3)
        zeta_v=np.ones((sys.o,1))*(-1)**np.random.randint(1,3)
        v[t]=np.dot(sys.V[t].G,zeta_v)+sys.V[t].x
        w[t]=np.dot(sys.W[t].G,zeta_w)+sys.W[t].x
        y[t]=np.dot(sys.C[t],x[t])+v[t]
        u[t]=np.zeros((sys.m,1))
        x[t+1]=np.dot(sys.A[t],x[t])+np.dot(sys.B[t],u[t])+w[t]  
    return x,y,u,v,w

import matplotlib.pyplot as plt
fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
y_minus,y_plus,u_minus,u_plus=[],[],[],[]
for t in range(T+1):
    y_minus.append(np.asscalar(S.ybar[t]-np.linalg.norm(S.phi[t],ord=np.inf)))
    y_plus.append(np.asscalar(S.ybar[t]+np.linalg.norm(S.phi[t],ord=np.inf)))
#ax.plot([0,T],[np.asscalar(Goal.x-Goal.G),np.asscalar(Goal.x-Goal.G)],'--',color='black')
#ax.plot([0,T],[np.asscalar(Goal.x+Goal.G),np.asscalar(Goal.x+Goal.G)],'--',color='black')
ax0.fill_between(range(T+1),y_minus,y_plus,alpha=0.3,color='red')
for t in range(T):
    u_minus.append(np.asscalar(S.ubar[t]-np.linalg.norm(S.theta[t],ord=np.inf)))
    u_plus.append(np.asscalar(S.ubar[t]+np.linalg.norm(S.theta[t],ord=np.inf)))
ax1.fill_between(range(T),u_minus,u_plus,color='green',alpha=0.4)
ax0.set_xlabel(r'time')
ax0.set_ylabel(r'$y$')
ax0.set_title(r'Reachable Outputs Over Time')
ax1.set_xlabel(r'time')
ax1.set_ylabel(r'$u$')
ax1.set_title(r'Possible Control Inputs Over Time')
ax2.set_xlabel(r'time')
ax2.set_ylabel(r'$x$')
ax2.set_title(r'Open-loop Output Response')
for i in range(1):
    zeta_x=2*(np.random.random((S.n,1))-0.5)
    x_0=np.dot(S.X0.G,zeta_x)+S.X0.x
    x,y,u,v,w=simulate(S,x_0,T)
    x2,y2,u2,v2,w2=simulate_zero(S,x_0,T)
    # X axis
    ax0.plot(range(T+1),[np.asscalar(y[t]) for t in range(T+1)],'o',color='red')
    ax0.plot(range(T+1),[np.asscalar(y[t]) for t in range(T+1)],'-',color='red')
    ax0.plot(range(T+1),[np.asscalar(y2[t]) for t in range(T+1)],'.',color='orange')
    ax0.plot(range(T+1),[np.asscalar(y2[t]) for t in range(T+1)],'--',color='orange')
    ax1.plot(range(T),[np.asscalar(u[t]) for t in range(T)],'o',color='green')
    ax1.plot(range(T),[np.asscalar(u[t]) for t in range(T)],'-',color='green')