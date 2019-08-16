#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from pypolytrajectory.LTV import system,test_controllability
from pypolytrajectory.reduction import reduced_order,order_reduction_error,error_construction,error_construction_old
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.zonotope_order_reduction.methods import G_cut,Girard
from pypolytrajectory.synthesis import output_feedback_synthesis,outputfeedback_synthesis_zonotope_solution,\
    triangular_stack_list,output_feedback_synthesis_lightweight_many_variables
from pypolytrajectory.system import LQG_LTV,LTV,LQG
import scipy.linalg as spa

# pypolycontain
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ

np.random.seed(0)
S=LTV()
n=5
m=1
o=1
z=2
T=42
S.X0=zonotope(np.ones((n,1))*100,np.eye(n)*1)
B=np.random.randint(0,2,size=(n,m))
B[0,0]=0
#B[1,0]=0
A=0.8*np.eye(n)+np.random.randint(-100,100,size=(n,n))*0.01*0.4
C=np.zeros((o,n))
C[0,0]=1
#C[1,1]=1
D=np.eye(n)[0:z,:]
for t in range(T):
    S.A[t]=A
    S.B[t]=B
    S.C[t]=C
    S.D[t]=D
    S.d[t]=np.zeros((z,1))
    S.W[t]=zonotope(np.zeros((n,1)),np.eye(n)*0.01)
    S.V[t]=zonotope(np.zeros((o,1)),np.eye(o)*0.01)
    S.QQ[t]=np.eye(n)*0
    S.RR[t]=np.eye(m)*1
    S.QQ[t][0,0]=1
S.F_cost=np.eye(n)*1

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



T=40

u_tilde,theta=output_feedback_synthesis_lightweight_many_variables(S,T=T)
#theta_zero={}
#u_tilde_zero={}
#for t in range(T):
#    theta_zero[t]=theta[t]*0
#    u_tilde_zero[t]=u_tilde[t]*0
Z,U=outputfeedback_synthesis_zonotope_solution(S,u_tilde,theta)
#Z,U=outputfeedback_synthesis_zonotope_solution(S,u_tilde_zero,theta_zero)


# Synthesis
Goal=zonotope(np.ones((1,1))*0,np.eye(1)*1)


    
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


def simulate_my_controller(sys,x_0,T,w,v):
    x,y,u,e,xi={},{},{},{},{}
    x[0]=x_0
    Y,U={},{}
    for t in range(T+1):
        print "simulating time:",t
        y[t]=np.dot(sys.C[t],x[t])+v[t]
        if t==T:
            return x,y,u,x
        if t==0:
            xi[0]=y[0]
        else:
            Y[t-1]=np.vstack([y[tau] for tau in range(t)])
            U[t-1]=np.vstack([u[tau] for tau in range(t)])
            e[t-1]=y[t]-np.dot(sys.M[t-1],Y[t-1])-np.dot(sys.N[t-1],U[t-1])
            xi[t]=np.vstack([y[0]]+[e[tau] for tau in range(t)])
        u[t]=u_tilde[t]+np.dot(theta[t],xi[t])
        x[t+1]=np.dot(sys.A[t],x[t])+np.dot(sys.B[t],u[t])+w[t]  

Z_red={}              
for t in range(T+1):
    G=Girard(Z[t].G,6)
    Z_red[t]=zonotope(Z[t].x,G,color="orange")
visZ([Z_red[t] for t in range(T+1)])    