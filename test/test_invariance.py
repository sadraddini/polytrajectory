#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:20:34 2019

@author: sadra
"""

import numpy as np
from pypolytrajectory.LTV import system,test_controllability
from pypolytrajectory.reduction import reduced_order,order_reduction_error,error_construction,error_construction_old
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.zonotope_order_reduction.methods import G_cut,Girard
from pypolytrajectory.synthesis import output_feedback_synthesis,outputfeedback_synthesis_zonotope_solution,\
    triangular_stack_list,output_feedback_synthesis_lightweight_many_variables,invariance_time
from pypolytrajectory.system import LQG_LTV,LTV,LQG
import scipy.linalg as spa

np.random.seed(0)
S=LTV()
n=6
m=2
o=1
z=1
T=22
S.X0=zonotope(np.ones((n,1))*0,np.eye(n)*1)
B=np.random.randint(0,2,size=(n,m))
B[0,0]=0
#B[1,0]=0
A=0.3*np.eye(n)+np.random.randint(-100,100,size=(n,n))*0.01*0.2
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
    S.A[t]=A
    S.B[t]=B
    S.C[t]=C
    S.D[t]=D
    S.d[t]=np.zeros((z,1))
    S.W[t]=zonotope(np.zeros((n,1)),np.eye(n)*0.01)
    S.V[t]=zonotope(np.zeros((o,1)),np.eye(o)*0.05)
    S.QQ[t]=np.eye(n)*0
    S.RR[t]=np.eye(m)*1
    S.QQ[t][0,0]=1
S.F_cost=np.eye(n)*0
S.F_cost[0,0]=1

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
plt.title(r"Location of Eigen-Values of $A$",fontsize=15)
plt.grid(lw=0.2,color=(0.2,0.3,0.2))
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()


import matplotlib.pyplot as plt0
plt0.plot(range(T-1),[np.linalg.norm(S.E[t].G,"fro") for t in range(T-1)],LineWidth=2,color='red')
plt0.plot(range(T-1),[np.linalg.norm(S.E[t].G,"fro") for t in range(T-1)],'o',MarkerSize=4,color='black')
plt0.title(r"$\mathbb{E}_t$ Over Time",fontsize=20)
plt0.xlabel(r"time",fontsize=20)
plt0.ylabel(r"$tr(EE')$",fontsize=20)
plt0.grid(lw=0.2,color=(0.2,0.3,0.2))

import matplotlib.pyplot as plt1
plt1.plot(range(T-1),[np.linalg.norm(S.F[t].G,"fro") for t in range(T-1)],LineWidth=3,color='green')
plt1.plot(range(T-1),[np.linalg.norm(S.F[t].G,"fro") for t in range(T-1)],'o',MarkerSize=4,color='black')
plt1.title(r"$\mathbb{F}_t$ Over Time",fontsize=20)
plt1.xlabel(r"time",fontsize=20)
plt1.ylabel(r"$tr(FF')$",fontsize=20)
plt1.grid(lw=0.2,color=(0.2,0.3,0.2))


def simulate_my_invariance_controller(sys,u_tilde,theta,x_0,T,w,v,N):
    """
        N is the number of episodes
    """
    x,y,u,e,xi={},{},{},{},{}
    x[0]=x_0
    Y,U={},{}
    for i in range(0,N):
        for t in range(T+1):
            print "simulating time:",t
            y[t]=np.dot(sys.C[t],x[i*T+t])+v[i*T+t]
            if t==T and i==N-1:
                return x,y,u
            elif t==T:
                break
            if t==0:
                xi[0]=y[0]
            else:
                Y[t-1]=np.vstack([y[tau] for tau in range(t)])
                U[t-1]=np.vstack([u[i*T+tau] for tau in range(t)])
                e[t-1]=y[t]-np.dot(sys.M[t-1],Y[t-1])-np.dot(sys.N[t-1],U[t-1])
                xi[t]=np.vstack([y[0]]+[e[tau] for tau in range(t)])
            u[i*T+t]=u_tilde[t]+np.dot(theta[t],xi[t])
            x[i*T+t+1]=np.dot(sys.A[t],x[i*T+t])+np.dot(sys.B[t],u[i*T+t])+w[i*T+t]  

   
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
            zeta_w=np.ones((sys.n,1))*(-1)**np.random.randint(1,3)
            zeta_v=np.ones((sys.o,1))*(-1)**np.random.randint(1,3)
            v[t]=np.dot(sys.V[0].G,zeta_v)+sys.V[0].x
            w[t]=np.dot(sys.W[0].G,zeta_w)+sys.W[0].x
    elif method=="zero":
        for t in range(T+1):
            zeta_w=np.zeros((sys.n,1))
            zeta_v=np.zeros((sys.o,1))
            v[t]=np.dot(sys.V[t].G,zeta_v)+sys.V[t].x
            w[t]=np.dot(sys.W[t].G,zeta_w)+sys.W[t].x
    else:
        raise NotImplementedError
    return w,v



T=10
H_U=zonotope(np.zeros((S.m,1)),np.eye(S.m)*10)
H_X=zonotope(np.zeros((S.n,1)),np.eye(S.n)*20)
u_tilde,theta,X,U=invariance_time(S,T,H_X,H_U) 
N=3
zeta_x=1*np.ones((S.n,1))*(-1)**np.random.randint(1,3)
x_0=np.dot(S.X0.G,zeta_x)+S.X0.x
w,v=generate_random_disturbance(S,T*N,method="extreme")
x,y,u=simulate_my_invariance_controller(S,u_tilde,theta,x_0,T,w,v,N)
# Plots
fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig0.set_size_inches(12, 8)
fig1.set_size_inches(12, 8)
for i in range(S.n):
    ax0.plot([x[t][i,0] for t in range(T*N)])               
for i in range(S.m): 
    ax1.plot([u[t][i,0] for t in range(T*N)])               
ax0.set_xlabel(r"t",FontSize=20)
ax1.set_xlabel(r"t",FontSize=20)
ax0.set_ylabel(r"x",FontSize=20)
ax1.set_ylabel(r"u",FontSize=20) 
for i in range(S.n):
    x_minus,x_plus=[],[]
    x_minus.append(np.asscalar(S.X0.x[i,0]-np.linalg.norm(S.X0.G[i,:],ord=1)))
    x_plus.append(np.asscalar(S.X0.x[i,0]+np.linalg.norm(S.X0.G[i,:],ord=1)))
    for t in range(1,T+1):
        x_minus.append(np.asscalar(X[t].x[i,0]-np.linalg.norm(X[t].G[i,:],ord=1)))
        x_plus.append(np.asscalar(X[t].x[i,0]+np.linalg.norm(X[t].G[i,:],ord=1)))
    ax0.fill_between(range(T+1),x_minus,x_plus,alpha=0.5,color='orange')
for i in range(S.m):
    u_minus,u_plus=[],[]
    for t in range(T):
        u_minus.append(np.asscalar(U[t].x[i,0]-np.linalg.norm(U[t].G[i,:],ord=1)))
        u_plus.append(np.asscalar(U[t].x[i,0]+np.linalg.norm(U[t].G[i,:],ord=1)))
    ax1.fill_between(range(T),u_minus,u_plus,color='purple',alpha=0.5)
    
#J=simulate_and_plot(N=1,disturbance_method="extreme",keys=["Our Method","TV-LQG","TI-LQG"])
#N=100
#J=simulate_and_cost_evaluate(N=N,disturbance_method="guassian",keys=["Our Method","TV-LQG","TI-LQG"])
#a=np.array([J[i]["Our Method"]/J[i]["TV-LQG"] for i in range(N)])
#b=np.array([J[i]["Our Method"]/J[i]["TI-LQG"] for i in range(N)])
#print np.mean(a),np.mean(b)