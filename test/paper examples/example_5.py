#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 23:51:39 2019

@author: sadra

The Compass gait
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


from pypolytrajectory.LTV import system,test_controllability
from pypolytrajectory.reduction import reduced_order,order_reduction_error,error_construction,error_construction_old
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.zonotope_order_reduction.methods import G_cut,Girard
from pypolytrajectory.synthesis import *
from pypolytrajectory.system import LQG_LTV,LTV,LQG
import scipy.linalg as spa

# pypolycontain
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes_ax as visZax

from scipy.spatial import ConvexHull
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pypolycontain.utils.utils import vertices_cube as vcube


def add_tube(ax,x,G,eps=0,list_of_dimensions=[0,1],axis=0):
    """
    Only when G is a zonotope generator
    """
    T=len(x.keys())
    p_list=[]
    n=x[0].shape[0]
    points_convex_all=np.empty((0,2))
    for t in range(0,T-1):
        y0=x[t].T+np.dot(G[t],vcube(G[t].shape[1]).T).T
        y1=x[t+1].T+np.dot(G[t+1],vcube(G[t+1].shape[1]).T).T
        y=np.vstack((y0,y1))
#        print y.shape
        points=y[:,list_of_dimensions]
        points_convex=points[ConvexHull(points).vertices,:]
#        print points_convex
        p=Polygon(points_convex)
        p_list.append(p)
        points_convex_all=np.vstack((points_convex_all,points_convex))
    p_patch = PatchCollection(p_list, color=(0.5,0.5,0.5),alpha=0.75)
    ax.add_collection(p_patch)
    ax.grid(color=(0,0,0), linestyle='--', linewidth=0.3)
    if axis==1:
        ax.set_xlim([min([x[t][0,0] for t in range(T)]),max([x[t][0,0] for t in range(T)])])
        ax.set_ylim([min([x[t][1,0] for t in range(T)]),max([x[t][1,0] for t in range(T)])])
    elif axis==0:
        ax.set_xlim([min(points_convex_all[:,0]),max(points_convex_all[:,0])])
        ax.set_ylim([min(points_convex_all[:,1]),max(points_convex_all[:,1])])
    return ax


#np.random.seed(0)

"""
Parameters
"""
m,m_h,a,b=2,2,0.5,0.5
n=4
S=LTV()
dt,T,g=0.04,45,10
l=a+b

M=np.array([[(m_h+m)*l**2+m*a**2,m*l*b],[m*l*b,m*b**2]])
Minv=np.linalg.pinv(M)
tau_g=g*np.array([[m_h*l+m*a+m*l,0],[0,-m*b]])



for t in range(T):
    S.A[t]=np.eye(n)
    S.A[t][0,2],S.A[t][1,3]=dt,dt
    S.A[t][2:,0:2]=np.dot(Minv,tau_g)*dt
    Bq=np.dot(Minv,np.array([1,1]))*dt
    S.B[t]=np.array([0,0,Bq[0],Bq[1]]).reshape(4,1)
#    S.B[t]=np.array([0,0,1,1]).reshape(4,1)
#    S.B[t]=np.array([[0,0,1,0],[0,0,0,1]]).T
#    S.B[t]=np.array([0,0,1,1]).reshape(4,1)
#    S.C[t]=np.array([[1,0,0,0],[0,1,0,0]]).reshape(2,4)
    S.C[t]=np.array([[1,1,0,0]]).reshape(1,4)
    S.D[t]=np.eye(n)
    S.d[t]=np.zeros((n,1))
    S.W[t]=zonotope(np.ones((n,1))*0,np.ones((n,1))*0)
#    S.V[t]=zonotope(np.zeros((2,1)),np.ones((2,1))*0.0000)
    S.V[t]=zonotope(np.zeros((1,1)),np.eye(1)*0.00)

if True:
    prog=MP.MathematicalProgram()
    T=20
    R=np.array([[0,-1,0,0],[-1,0,0,0],[0,0,0,-1],[0,0,-1,0]])
    x,u={},{}
    for t in range(T+1):
        x[t]=prog.NewContinuousVariables(4,1,"x%d"%t)
        u[t]=prog.NewContinuousVariables(1,1,"u%d"%t)
    x[0][0,0]=0.12
    x[0][1,0]=0.12
    for t in range(T):
        x_new=np.dot(S.A[0],x[t])+np.dot(S.B[0],u[t])
        prog.AddLinearConstraint(np.equal(x[t+1],x_new,dtype='object').flatten())
    prog.AddLinearConstraint(np.equal(x[0],np.dot(R,x[T]),dtype='object').flatten())
    J=0
    for t in range(T):
        J+=sum(u[t].flatten()**2)
    prog.AddQuadraticCost(J)
    result=gurobi_solver.Solve(prog,None,None)
    if result.is_success():
        print("Trajectory Optimization Success!","\n"*2)
        x_n={t:result.GetSolution(x[t]) for t in range(0,T+1)}
        u_n={t:result.GetSolution(u[t]) for t in range(0,T)}
    

# Initial Condition
x0_bar=sym.Evaluate(x_n[0]).reshape(4,1)
x_n[0]=x0_bar.reshape(4)
G=0.0001*np.eye(4)
G[2,2]=0.0001
G[3,3]=0.0001
S.X0=zonotope(x0_bar,G)

plt.plot([x_n[t][0] for t in range(T+1)],[x_n[t][2] for t in range(T+1)],color='red')
plt.plot([x_n[t][0] for t in range(T+1)],[x_n[t][2] for t in range(T+1)],'o',color='red')
plt.plot([x_n[t][1] for t in range(T+1)],[x_n[t][3] for t in range(T+1)],color='blue')
plt.plot([x_n[t][1] for t in range(T+1)],[x_n[t][3] for t in range(T+1)],'o',color='blue')

#raise 1
S.construct_dimensions()
S._abstract_evolution_matrices()
S._error_zonotopes()

    
def simulate_my_invariance_controller(sys,u_tilde,theta,x_0,T,w,v,N):
    """
        N is the number of episodes
    """
    x,y,u,e,xi={},{},{},{},{}
    x[0]=x_0
    Y,U={},{}
    for i in range(0,N):
        for t in range(T+1):
            print("simulating time:",t)
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
            zeta_w,zeta_v=2*(np.random.random((sys.W[t].G.shape[1],1))-0.5),2*(np.random.random((sys.V[t].G.shape[1],1))-0.5)
            v[t]=np.dot(sys.V[t].G,zeta_v)+sys.V[t].x
            w[t]=np.dot(sys.W[t].G,zeta_w)+sys.W[t].x
    elif method=="extreme":
        for t in range(T+1):
            zeta_w=np.ones((sys.W[t].G.shape[1],1))*(-1)**np.random.randint(1,3)
            zeta_v=np.ones((sys.V[t].G.shape[1],1))*(-1)**np.random.randint(1,3)
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

def visualize(ax,x):
    ax.plot([0,-l*np.sin(x[0])],[0,l*np.cos(x[0])],LineWidth=5,color='red')
    ax.plot([0,-a*np.sin(x[0])],[0,a*np.cos(x[0])],'o',MarkerSize=5,color='blue')
    ax.plot([-l*np.sin(x[0]),-l*np.sin(x[0])-l*np.sin(x[1])],[l*np.cos(x[0]),l*np.cos(x[0])-l*np.cos(x[1])],LineWidth=5,color='red')
    ax.plot([-l*np.sin(x[0]),-l*np.sin(x[0])-a*np.sin(x[1])],[l*np.cos(x[0]),l*np.cos(x[0])-a*np.cos(x[1])],'o',MarkerSize=5,color='blue')
    
#T=14
#H_U=zonotope(np.zeros((1,1)),np.eye(1)*1000)
#G_X=np.eye(4)*10000
#G_X[0,0]=0.3
#G_X[1,1]=0.3
#H_X=zonotope(np.zeros((S.n,1)),G_X)
#R=np.eye(4)
u_tilde,theta,X,U=invariance_time(S,T,reset_map=R,H_X=None,H_U=None) 
N=2
zeta_x=0*np.ones((S.X0.G.shape[1],1))*(-1)**np.random.randint(1,3)
x_0=np.dot(S.X0.G,zeta_x)+S.X0.x
w,v=generate_random_disturbance(S,T*N,method="extreme")
x,y,u=simulate_my_invariance_controller(S,u_tilde,theta,x_0,T,w,v,N)

fig,ax = plt.subplots()   
fig.set_size_inches(8, 8) 
ax.plot([x[t][0] for t in range(T+1)],[x[t][2] for t in range(T+1)],color='red',label=r'Stance')
ax.plot([x[t][0] for t in range(T+1)],[x[t][2] for t in range(T+1)],'o',color='red')
ax.plot([x[t][1] for t in range(T+1)],[x[t][3] for t in range(T+1)],color='blue',label=r'Swing')
ax.plot([x[t][1] for t in range(T+1)],[x[t][3] for t in range(T+1)],'o',color='blue')

X_red={}    
      
for t in range(1,T+1):
    G=Girard(X[t].G,10)
    X_red[t]=zonotope(X[t].x,G,color='red')
    X_red[t].G=X_red[t].G

X_red[0]=S.X0
    
visZax(ax,[X_red[t] for t in range(0,T+1)],alpha=0.8,a=0.1,list_of_dimensions=[0,2])   
x_tube={t:X_red[t].x for t in range(0,T+1)}
G_tube={t:X_red[t].G for t in range(0,T+1)}
x_tube[T]=np.dot(R,x_tube[T])
G_tube[T]=np.dot(R,G_tube[T])
add_tube(ax,x_tube,G_tube,eps=0,list_of_dimensions=[0,2],axis=0)

for t in range(1,T+1):
    G=Girard(X[t].G,10)
    X_red[t]=zonotope(X[t].x,G,color='blue')
    X_red[t].G=X_red[t].G
    
visZax(ax,[X_red[t] for t in range(0,T+1)],alpha=0.8,a=0.1,list_of_dimensions=[1,3])   
add_tube(ax,x_tube,G_tube,eps=0,list_of_dimensions=[1,3],axis=0)
ax.set_xlim([-0.18,0.18])
ax.set_ylim([-1.2,1.2])
ax.set_title(r"Compass Gait",FontSize=20)
ax.set_xlabel(r"$\theta$",FontSize=20)
ax.set_ylabel(r"$\dot{\theta}$",FontSize=20)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels,fontsize=20)
fig.savefig('figures/Example5', dpi=100)


#fig,ax = plt.subplots()
#fig.set_size_inches(10, 10)
#for t in range(1,T+1):
#    fig,ax = plt.subplots()
#    fig.set_size_inches(5, 5)
#    visualize(ax,x_n[t])
    