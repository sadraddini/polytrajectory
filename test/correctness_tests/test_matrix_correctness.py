#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:30:34 2019

@author: sadra
"""

import numpy as np
from pypolytrajectory.LTV import system,test_controllability
from pypolytrajectory.reduction import reduced_order,order_reduction_error,error_construction,error_construction_old
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.zonotope_order_reduction.methods import G_cut,Girard
from pypolytrajectory.synthesis import output_feedback_synthesis,outputfeedback_synthesis_zonotope_solution,output_feedback_synthesis_lightweight,\
    triangular_stack_list,output_feedback_synthesis_lightweight_many_variables
from pypolytrajectory.system import LQG_LTV,LTV,LQG
import scipy.linalg as spa

np.random.seed(0)
 
S=LTV()
n=3
m=1
o=1
z=3
T=58
S.X0=zonotope(np.ones((n,1))*0,np.eye(n)*1)
B=np.random.randint(0,2,size=(n,m))
B[0,0]=1
B[1,0]=0
A=0.9*np.eye(n)+np.random.randint(-100,100,size=(n,n))*0.01*0.15
C=np.zeros((o,n))
C[0,0]=1
#C[1,1]=1
D=np.eye(n)[0:z,:]
#A=np.array([[0.28,0.25,-0.19,-0.22,0.03,-0.50],
#                 [0.25,-0.47,0.30,0.17,-0.11,-0.11],
#                 [-0.19,0.30,0.46,0.09,-0.02,-0.08],
#                 [-0.22,0.17,0.09,0.60,-0.06,0.14],
#                 [0.03,-0.11,-0.02,-0.06,0.46,-0.13],
#                 [-0.50,-0.11,-0.08,0.14,-0.13,-0.23]]).reshape(6,6)
#B=np.array([[1.0159,0,0.5988,1.8641,0,-1.2155]]).reshape(6,1)
#C=np.array([[1.292,0,0,0.2361,0.8428,0]]).reshape(1,6)
for t in range(T):
    S.A[t]=0.9*np.eye(n)+np.random.randint(-100,100,size=(n,n))*0.01*0.15
    S.B[t]=np.random.randint(0,2,size=(n,m))
    S.C[t]=C
    S.D[t]=D
    S.d[t]=np.zeros((z,1))
    S.W[t]=zonotope(np.zeros((n,1)),np.eye(n)*0.05)
    S.V[t]=zonotope(np.zeros((o,1)),np.eye(o)*0.05)
    S.QQ[t]=np.eye(n)*1
    S.RR[t]=np.eye(m)*1
S.F_cost=np.eye(n)*0.00

S.U_set=zonotope(np.zeros((m,1)),np.eye(m)*1)
S.construct_dimensions()
#S.construct_E()
S._abstract_evolution_matrices()
S._error_zonotopes()



import matplotlib.pyplot as plt
L=np.linalg.eigvals(S.A[0])
X = [l.real for l in L]
Y = [l.imag for l in L]
plt.scatter(X,Y, color='red',s=50)
N=200
x,y=[np.cos(2*np.pi/N*i) for i in range(N)],[np.sin(2*np.pi/N*i) for i in range(N)]
plt.plot(x,y,color='black')
plt.axis("equal")
plt.title(r"Location of Open-loop Poles")
plt.show()


import matplotlib.pyplot as plt0
plt0.plot(range(T-1),[np.linalg.norm(S.E[t].G[0,:],1) for t in range(T-1)],LineWidth=2,color='orange')
plt0.plot(range(T-1),[np.linalg.norm(S.E[t].G[0,:],1) for t in range(T-1)],'o',MarkerSize=4,color='orange')
plt0.title(r"Error Over Time")

import matplotlib.pyplot as plt0
plt0.plot(range(T-1),[np.linalg.norm(S.F[t].G,"fro") for t in range(T-1)],LineWidth=3,color='red')
plt0.plot(range(T-1),[np.linalg.norm(S.F[t].G,"fro") for t in range(T-1)],'o',MarkerSize=4,color='black')
plt0.title(r"Error Over Time")
plt0.grid(lw=0.2,color=(0.2,0.3,0.2))

def generate_random_disturbance(sys,T,method="guassian"):
    w,v={},{}
    if method=="guassian":
        for t in range(T+1):
            w[t]=np.random.multivariate_normal(mean=sys.W[t].x.reshape(sys.n),cov=sys.W[t].G).reshape(sys.n,1)
            v[t]=np.random.multivariate_normal(mean=sys.V[t].x.reshape(sys.o),cov=sys.V[t].G).reshape(sys.o,1)
    elif method=="uniform":
         for t in range(T+1):
            zeta_w,zeta_v=2*(np.random.random((sys.n,1))-0.5),2*(np.random.random((sys.o,1))-0.5)
            v[t]=np.dot(sys.V[t].G,zeta_v)+sys.V[t].x
            w[t]=np.dot(sys.W[t].G,zeta_w)+sys.W[t].x
    elif method=="extreme":
        for t in range(T+1):
            zeta_w=np.ones((sys.n,1))*(-1)**np.random.randint(1,2)
            zeta_v=np.ones((sys.o,1))*(-1)**np.random.randint(1,2)
            v[t]=np.dot(sys.V[t].G,zeta_v)+sys.V[t].x
            w[t]=np.dot(sys.W[t].G,zeta_w)+sys.W[t].x
    elif method=="zero":
        for t in range(T+1):
            zeta_w=np.zeros((sys.n,1))
            zeta_v=np.zeros((sys.o,1))
            v[t]=np.dot(sys.V[t].G,zeta_v)+sys.V[t].x
            w[t]=np.dot(sys.W[t].G,zeta_w)+sys.W[t].x
    else:
        raise NotImplementedError
    return w,v


def simulate_random(sys,x_0,T,w,v):
    x,y,u,x_observer={},{},{},{}
    x[0]=x_0
    y[0]=np.dot(sys.C[0],x[0])+v[0]    
    for t in range(T+1):
#        print "simulating observer time:",t
        if t==T:
            return x,y,u
        u[t]=np.dot(np.random.random((sys.m,sys.n))-0.5,x[t])
        x[t+1]=np.dot(sys.A[t],x[t])+np.dot(sys.B[t],u[t])+w[t]
        y[t+1]=np.dot(sys.C[t+1],x[t+1])+v[t+1]
#        print "control",t,u[t],u[t].shape

T=30        
x_0=np.random.random((S.n,1))
w,v=generate_random_disturbance(S,T,"extreme")
x,y,u=simulate_random(S,x_0,T,w,v)

print "Now the check time"
for t in range(T):
    print t
    Y=np.vstack([y[tau] for tau in range(t+1)])
    U=np.vstack([u[tau] for tau in range(t+1)])
    W=np.vstack([w[tau] for tau in range(t+1)])
    V=np.vstack([v[tau] for tau in range(t+1)])
#    print S.Q["x",t].shape, S.Q["w",t].shape,  S.Q["v",t].shape,  S.Q["u",t].shape, 
    error_Y= Y-( np.dot(S.Q["x",t],x_0) + np.dot(S.Q["u",t],U) + np.dot(S.Q["w",t],W) + np.dot(S.Q["v",t],V) )
    print "Y",t, np.linalg.norm(error_Y)
#    print "Y",error_Y
    if t>=1:
        U=np.vstack([u[tau] for tau in range(t)])
        W=np.vstack([w[tau] for tau in range(t)])
        error_X= x[t]-( np.dot(S.P["x",t],x_0) + np.dot(S.P["u",t],U) + np.dot(S.P["w",t],W))
        print "x",t, np.linalg.norm(error_X)
    