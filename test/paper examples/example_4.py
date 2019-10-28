#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 01:55:51 2019

@author: sadra

Example 4: Output Feedback Motion Planning

The robust game
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

#np.random.seed(0)

"""
Environment Generation
"""
K=10
n=2*K+3
v,dt,g,d_0=1,0.429,-10,1
S=LTV()
T=int((K+d_0)/(v*dt))+1

gap_bar,step_var,gap_var=1,0.3,0.1
x0_bar=np.zeros((n,1))
G=np.zeros((n,n))
x0_bar[0,0],x0_bar[1,0]=-d_0,0
G[0,0]=0 # Variation in x
G[1,1]=0 # Variation in speed
G[2,2]=0 # Variation in height
for i in range(K):
    x0_bar[3+2*i+1]=gap_bar
    G[3+2*i+1,3+2*i+1]=gap_var
    for j in range(K-i):
        G[3+2*i+2*j,3+2*i]=step_var
      
S.X0=zonotope(x0_bar,G)

def visualize(ax,x):
    # Add the patch to the Axes
    for i in range(K):
        rect_down = patches.Rectangle((i,x[3+2*i]-10),1,10,linewidth=1,edgecolor='black',facecolor='orange',alpha=0.59)
        rect_up = patches.Rectangle((i,x[3+2*i]+x[3+2*i+1]),1,10,linewidth=1,edgecolor='black',facecolor='orange',alpha=0.59)
        ax.set_xlim([-1,K-1])
        ax.set_ylim([-3,3])
        ax.add_patch(rect_down)
        ax.add_patch(rect_up)
        
#zeta=2*(np.random.random((n,1))-0.5)
#x=np.dot(S.X0.G,zeta)+S.X0.x
#visualize(x)


        
# Dynamics
for t in range(T):
    S.A[t]=np.eye(n)    
    S.B[t]=np.zeros((n,1))
    S.A[t][1,2],S.B[t][2,0],S.B[t][1,0]=dt,dt,dt**2/2.0
    S.W[t]=zonotope(np.zeros((n,1)),np.eye(n)*0)
    S.W[t].G[1,1]=0.01
    S.W[t].x[1,0],S.W[t].x[2,0]=g*dt**2/2.0,g*dt
    S.W[t].x[0,0]=v*dt
# Observation
N=2
for t in range(T):
    S.C[t]=np.zeros((2*N,n))
    x_now=-d_0+v*t*dt
    k=max(0,int(x_now))
    for j in range(N):
        if 3+2*k+1+2*j<=n:
            S.C[t][2*j,3+2*k+2*j]=-1
            S.C[t][2*j,1]=1
            S.C[t][2*j+1,3+2*k+1+2*j]=1
    S.V[t]=zonotope(np.zeros((2*N,1)),np.eye(2*N)*0.02)
    S.D[t]=np.zeros((2,n))
    if x_now>0:
        S.D[t][0,1],S.D[t][0,3+2*k]=1,-1
        S.D[t][1,1],S.D[t][1,3+2*k],S.D[t][1,3+2*k+1]=-1,1,1
    S.d[t]=np.zeros((2,1))

S.construct_dimensions()
S._abstract_evolution_matrices()
S._error_zonotopes()

H_U={}
H_Z={}
D=5
T_synthesize=T-2

for t in range(T_synthesize):
    H_U[t]=zonotope(x=np.array([0]).reshape(1,1),G=np.eye(1)*2*g,color=(0,0.0,1))
    H_Z[t]=zonotope(x=np.array([D,D]).reshape(2,1),G=np.eye(2)*D,color=(1,0.8,0.8))
    
u_tilde,theta=output_feedback_synthesis_lightweight_many_variables(S,T_synthesize,H_Z=H_Z,H_U=H_U)
#theta_zero={}
#u_tilde_zero={}
#for t in range(T):
#    theta_zero[t]=theta[t]*0
#    u_tilde_zero[t]=u_tilde[t]*0
Z,U=outputfeedback_synthesis_zonotope_solution(S,u_tilde,theta)

# Simulation
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

# Simulate
w,v=generate_random_disturbance(S,T_synthesize,"extreme")
zeta=2*(np.random.random((n,1))-0.5)
zeta=(-0.99*np.ones((S.n,1)))**np.random.randint(1,3,size=((S.n,1)))
x0=np.dot(S.X0.G,zeta)+S.X0.x
x,y,u=simulate_my_controller(S,x0,T_synthesize,w,v)      
# Create figure and axes
fig,ax = plt.subplots()
fig.set_size_inches(7, 2)  
visualize(ax,x[0])
z={}
for t in range(T_synthesize):
    ax.plot(x[t][0,0],x[t][1,0],'o',MarkerSize=6,color='blue')
    ax.plot([x[tau][0,0] for tau in range(t+1)],\
             [x[tau][1,0] for tau in range(t+1)],\
             '--',LineWidth=1,color='black')
    z[t]=np.dot(S.D[t],x[t])

#fig2, ax2 = plt.subplots() # note we must use plt.subplots, not plt.subplot
#Z_red={}       
#fig2.set_size_inches(12, 12)       
#for t in range(T_synthesize+1):
#    G=Girard(Z[t].G,10)
#    Z_red[t]=zonotope(Z[t].x,G,color=(t/(T+10.1),0.9,0.5))
#    Z_red[t].G+=np.random.random((2,10))*0.00001
        
        
#for Tau in range(T_synthesize):
#    visZax(ax2,[H_Z[T_synthesize-1]]+[Z_red[t] for t in range(Tau)]+[zonotope(Z_red[Tau].x,Z_red[Tau].G,color='cyan')],alpha=0.8)    
#    #visZax(ax,[H_Z[T]]+[Z_red[T]],alpha=0.99)   
#    ax2.set_title(r"Performance Variables",fontsize=26) 
#    ax2.set_xlabel(r"$z_1$",fontsize=26) 
#    ax2.set_ylabel(r"$z_2$",fontsize=26) 
#    ax2.set_xlim([-0.1,1])
#    ax2.set_ylim([-0.1,2.5])
#    ax2.plot([Z_red[t].x[0,0] for t in range(Tau+1)],[Z_red[t].x[1,0] for t in range(Tau+1)],'-',Linewidth=3,color='blue') 
#    ax2.plot([Z_red[t].x[0,0] for t in range(Tau+1)],[Z_red[t].x[1,0] for t in range(Tau+1)],'o',MarkerSize=5,color='blue')
#    fig2.savefig('figures/game_z_%d.png'%Tau, dpi=100)