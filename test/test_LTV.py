#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:59:32 2019

@author: sadra
"""
import numpy as np
from pypolytrajectory.LTV import system
from pypolytrajectory.reduction import reduced_order
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.zonotope_order_reduction.methods import G_cut
from pypolytrajectory.synthesis import synthesis,zonotopic_controller

S=system()
n=3
m=1
o=1
T=20
#np.random.seed(1)
S.X0=zonotope(np.array(([0,0,0])).reshape(3,1),np.eye(n)*1)
for t in range(T):
    S.A[t]=np.array([[1,0.2,-0.1],[-0.1,1,-0.1],[0.0,0.1,0.8]]).reshape(3,3)
    S.B[t]=np.array([[0,0.2,0.0]]).reshape(3,1)
    S.C[t]=np.array([[1,0,0]]).reshape(1,3)
    S.W[t]=zonotope(np.zeros((n,1)),np.eye(n)*0.003)
    S.V[t]=zonotope(np.zeros((o,1)),np.eye(o)*0.001)

S.U_set=zonotope(np.zeros((m,1)),np.eye(m)*2)
S.construct_dimensions()
S.construct_E()
M,N,Z=reduced_order(S,T-1)

for t in range(T-1):
    print t,"-zonotope reduction"
    D,G=G_cut(Z[t],4,'osqp')
    Z[t].G=G

S.Z=Z
S.M=M
S.N=N

# Synthesis
T=17
Goal=zonotope(np.ones((1,1))*0,np.eye(1)*1)    
synthesis(S,q0=3,T=T,y_goal=Goal)

def simulate(sys,x_0,T):
    x,y,u,v,w={},{},{},{},{}
    x[0]=x_0
    for t in range(T+1):
        print "simulating time:",t
        zeta_w,zeta_v=2*(np.random.random((sys.n,1))-0.5),2*(np.random.random((sys.o,1))-0.5)
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

zeta_x=2*(np.random.random((S.n,1))-0.5)
x_0=np.dot(S.X0.G,zeta_x)+S.X0.x
x_0[0,0]=S.ybar[0]+S.Phi[0][0,0]*0.9
x_0[1,0]=-3
x_0[2,0]=1
x,y,u,v,w=simulate(S,x_0,T)

# X axis
import matplotlib.pyplot as plt
y_minus,y_plus,u_minus,u_plus=[],[],[],[]
for t in range(T+1):
    y_minus.append(np.asscalar(S.ybar[t]-np.linalg.norm(S.Phi[t][-1,:],ord=1)))
    y_plus.append(np.asscalar(S.ybar[t]+np.linalg.norm(S.Phi[t][-1,:],ord=1)))
plt.fill_between(range(T+1),y_minus,y_plus,alpha=0.3,color='red')
plt.plot(range(T+1),[np.asscalar(y[t]) for t in range(T+1)],'o',color='red')
plt.plot(range(T+1),[np.asscalar(y[t]) for t in range(T+1)],'-',color='red')
for t in range(T):
    u_minus.append(np.asscalar(S.ubar[t]-np.linalg.norm(S.Theta[t][-1,:],ord=1)))
    u_plus.append(np.asscalar(S.ubar[t]+np.linalg.norm(S.Theta[t][-1,:],ord=1)))
plt.plot([0,T],[np.asscalar(Goal.x-Goal.G),np.asscalar(Goal.x-Goal.G)],'--',color='black')
plt.plot([0,T],[np.asscalar(Goal.x+Goal.G),np.asscalar(Goal.x+Goal.G)],'--',color='black')
plt.xlabel('time')
plt.ylabel('y')
plt.title('Reachable Outputs Over Time')
#plt.fill_between(range(T),u_minus,u_plus,color='red')
