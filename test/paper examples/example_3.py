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
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes_ax as visZax

np.random.seed(0)
S=LTV()
n=10
m=2
o=1
z=2
T=42
x0_bar=np.ones((n,1))*10
x0_bar[0,0]=25
x0_bar[1,0]=-25
S.X0=zonotope(x0_bar,np.eye(n)*1)
B=np.random.randint(0,2,size=(n,m))*1
#B[0,0]=0.05
#B[1,1]=0.05
A=0.95*np.eye(n)+np.random.randint(-100,100,size=(n,n))*0.01*0.1
#n_2=8
#A[n-n_2:,n-n_2:]=np.eye(n_2)
#A[n-n_2:,:n-n_2]=np.zeros((n_2,n-n_2))
for i in range(n):
    for j in range(n):
        if i>j:
            A[i,j]=0 
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
    S.W[t]=zonotope(np.ones((n,1))*0,np.eye(n)*0.01)
    S.V[t]=zonotope(np.zeros((o,1)),np.eye(o)*0.05)
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



T=20
H_U={}
H_Z={}
for t in range(T):
    H_U[t]=zonotope(x=np.array([0,0]).reshape(2,1),G=np.eye(2)*3,color=(0,0.0,1))
#    H_Z[t]=zonotope(x=np.array([0,0]).reshape(2,1),G=np.eye(2)*25,color=(1,0.9,0.9))
H_Z[T]=zonotope(x=np.array([5,5]).reshape(2,1),G=1*np.array([[5,0],[0,10]]),color=(1,0.0,0.0))
    
u_tilde,theta=output_feedback_synthesis_lightweight_many_variables(S,T=T,H_Z=H_Z,H_U=H_U)
#theta_zero={}
#u_tilde_zero={}
#for t in range(T):
#    theta_zero[t]=theta[t]*0
#    u_tilde_zero[t]=u_tilde[t]*0
Z,U=outputfeedback_synthesis_zonotope_solution(S,u_tilde,theta)
#Z,U=outputfeedback_synthesis_zonotope_solution(S,u_tilde_zero,theta_zero)


    
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
        y[t]=np.dot(sys.C[t],x[t])+v[t]
        if t==T:
            return x,y,u
        if t==0:
            xi[0]=y[0]
        else:
            Y[t-1]=np.vstack([y[tau] for tau in range(t)])
            U[t-1]=np.vstack([u[tau] for tau in range(t)])
            e[t-1]=y[t]-np.dot(sys.M[t-1],Y[t-1])-np.dot(sys.N[t-1],U[t-1])
            xi[t]=np.vstack([y[0]]+[e[tau] for tau in range(t)])
        u[t]=u_tilde[t]+np.dot(theta[t],xi[t])
        x[t+1]=np.dot(sys.A[t],x[t])+np.dot(sys.B[t],u[t])+w[t] 

w,v=generate_random_disturbance(S,T,"extreme")
x0=0.9*np.ones((S.n,1))*(-1)**np.random.randint(1,3)+S.X0.x
x,y,u=simulate_my_controller(S,x0,T,w,v)        
fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
Z_red={}       
fig.set_size_inches(12, 12)       
for t in range(T+1):
    G=Girard(Z[t].G,10)
    Z_red[t]=zonotope(Z[t].x,G,color=(t/(T+10.1),t/(T+10.1),t/(T+10.1)))
U_red={}       
for t in range(T):
    G=Girard(U[t].G,10)
    U_red[t]=zonotope(U[t].x,G+np.random.random(G.shape)*0.01,color=(t/(T+10.1),t/(T+10.1),t/(T+0.1)))
        
        
for Tau in range(T+1):
    fig,ax = plt.subplots()
    fig.set_size_inches(10, 10)
    visZax(ax,[H_Z[T]]+[Z_red[t] for t in range(Tau)]+[zonotope(Z_red[Tau].x,Z_red[Tau].G,color='cyan')],alpha=0.8)    
    #visZax(ax,[H_Z[T]]+[Z_red[T]],alpha=0.99)   
    ax.set_title(r"Trajectory of Performance Variables",fontsize=26) 
    ax.set_xlabel(r"$x_1$",fontsize=26) 
    ax.set_ylabel(r"$x_2$",fontsize=26) 
    ax.set_xlim([-10,30])
    ax.set_ylim([-30,30])
    ax.plot([x[t][0,0] for t in range(Tau+1)],[x[t][1,0] for t in range(Tau+1)],'-',Linewidth=3,color='blue') 
    ax.plot([x[t][0,0] for t in range(Tau+1)],[x[t][1,0] for t in range(Tau+1)],'o',MarkerSize=5,color='blue')
    fig.savefig('figures/Example3_mpc_z_%d.png'%Tau, dpi=100)

for Tau in range(T):
    fig2, ax2 = plt.subplots() # note we must use plt.subplots, not plt.subplot
    fig2.set_size_inches(12, 12)       
    visZax(ax2,[H_U[T-1]]+[U_red[t] for t in range(Tau+1)]+[zonotope(U_red[Tau].x,U_red[Tau].G,color='cyan')],alpha=0.5)  
    ax2.set_title(r"Control Inputs",fontsize=26) 
    ax2.set_xlabel(r"$u_1$",fontsize=26) 
    ax2.set_ylabel(r"$u_2$",fontsize=26) 
    ax2.plot([u[t][0,0] for t in range(Tau+1)],[u[t][1,0] for t in range(Tau+1)],'-',Linewidth=3,color=(0,0.3,0))  
    ax2.plot([u[t][0,0] for t in range(Tau+1)],[u[t][1,0] for t in range(Tau+1)],'o',MarkerSize=5,color=(0,0.3,0))  
    fig2.savefig('figures/Example3_mpc_u_%d.png'%Tau, dpi=100)