#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:58:14 2019

@author: sadra
"""


import numpy as np
from pypolytrajectory.LTV import system,test_controllability
from pypolytrajectory.reduction import reduced_order,order_reduction_error,error_construction,error_construction_old
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.zonotope_order_reduction.methods import G_cut,Girard
from pypolytrajectory.synthesis import output_feedback_synthesis,outputfeedback_synthesis_zonotope_solution,\
    triangular_stack_list,output_feedback_synthesis_lightweight_many_variables
from pypolytrajectory.system import LQG_LTV,LTV,LQG
import scipy.linalg as spa

np.random.seed(1000)
S=LTV()
n=6
m=1
o=1
z=1
T=42
S.X0=zonotope(np.ones((n,1))*0,np.eye(n)*1)
B=np.random.randint(0,2,size=(n,m))
B[0,0]=0
A=0.0*np.eye(n)+np.random.randint(-100,100,size=(n,n))*0.01
C=np.zeros((o,n))
C[0,0]=1
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

S.construct_dimensions()
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
plt0.title(r"Observability Error Over Time",fontsize=20)
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


def simulate_LQG_LTV(sys,x_0,T,w,v):
    x,y,u,x_observer={},{},{},{}
    x[0]=x_0
    y[0]=np.dot(sys.C[0],x[0])+v[0]    
    x_observer[0]=np.dot(np.linalg.pinv(S.C[0]),y[0])
    x_observer[0]=sys.X0.x
    for t in range(T+1):
#        print "simulating observer time:",t
        if t==T:
            return x,y,u,x_observer
        u[t]=np.dot(-S.K[t],x_observer[t])
        x[t+1]=np.dot(sys.A[t],x[t])+np.dot(sys.B[t],u[t])+w[t]
        y[t+1]=np.dot(sys.C[t+1],x[t+1])+v[t+1]
        x_observer[t+1]=np.dot(sys.A[t],x_observer[t])+np.dot(sys.B[t],u[t])+\
            np.dot(sys.L[t+1],y[t+1]-np.dot(sys.C[t+1],np.dot(sys.A[t],x_observer[t])+np.dot(sys.B[t],u[t])))
#        print "control",t,u[t],u[t].shape
            
def simulate_LQG_LTI(sys,x_0,T,w,v,L,K):
    x,y,u,x_observer={},{},{},{}
    x[0]=x_0
    y[0]=np.dot(sys.C[0],x[0])+v[0]    
    x_observer[0]=np.dot(np.linalg.pinv(S.C[0]),y[0])
    x_observer[0]=sys.X0.x
    for t in range(T+1):
#        print "simulating observer time:",t
        if t==T:
            return x,y,u,x_observer
        u[t]=np.dot(-K,x_observer[t])
        x[t+1]=np.dot(sys.A[t],x[t])+np.dot(sys.B[t],u[t])+w[t]
        y[t+1]=np.dot(sys.C[t+1],x[t+1])+v[t+1]
        x_observer[t+1]=np.dot(sys.A[t],x_observer[t])+np.dot(sys.B[t],u[t])+\
            np.dot(L,y[t+1]-np.dot(sys.C[t+1],np.dot(sys.A[t],x_observer[t])+np.dot(sys.B[t],u[t])))
#        print "control",t,u[t],u[t].shape

def simulate_my_controller(sys,x_0,T,w,v):
    x,y,u,e,xi={},{},{},{},{}
    x[0]=x_0
    Y,U={},{}
    for t in range(T+1):
        print("simulating time:",t)
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
        
    
def simulate_and_cost_evaluate(N=1,disturbance_method="extreme",keys=["Our Method","TV-LQG"]):
    J={}
    if "TI-LQG" in keys:
        L,K=LQG(S.A[0],S.B[0],S.C[0],S.W[0].G,S.V[0].G,S.QQ[0],S.RR[0])
    if "TV-LQG" in keys:
        S.L,S.K=LQG_LTV(S,T)
    for i in range(N):
        print("iteration",i)
        J[i]={}
        zeta_x=2*(np.random.random((S.n,1))-0.5)
        x_0=np.dot(S.X0.G,zeta_x)+S.X0.x
        w,v=generate_random_disturbance(S,T,method=disturbance_method)
        x,y,u,x_observer={},{},{},{}
        for method in keys:
            if method=="Our Method":
                x[method],y[method],u[method],x_observer[method]=simulate_my_controller(S,x_0,T,w,v)
            elif method=="TI-LQG":
                x[method],y[method],u[method],x_observer[method]=simulate_LQG_LTI(S,x_0,T,w,v,L,K)
            elif method=="TV-LQG":
                x[method],y[method],u[method],x_observer[method]=simulate_LQG_LTV(S,x_0,T,w,v)
        for method in x.keys():
            J[i][method]=sum([np.linalg.norm(np.dot(S.D[t],x[method][t]),ord=2)**2+np.linalg.norm(u[method][t],ord=2)**2 for t in range(T)])
    return J
        
        
def simulate_and_plot(N=1,disturbance_method="extreme",keys=["Our Method","TV-LQG"]):
    import matplotlib.pyplot as plt
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
#    fig2, ax2 = plt.subplots()
    fig0.set_size_inches(12, 8)
    fig1.set_size_inches(12, 8)
    y_minus,y_plus,u_minus,u_plus=[],[],[],[]
    for t in range(T+1):
        y_minus.append(np.asscalar(Z[t].x[0,0]-np.linalg.norm(Z[t].G[0,:],ord=1)))
        y_plus.append(np.asscalar(Z[t].x[0,0]+np.linalg.norm(Z[t].G[0,:],ord=1)))
    ax0.fill_between(range(T+1),y_minus,y_plus,alpha=0.5,color='orange')
    y_minus,y_plus,u_minus,u_plus=[],[],[],[]
    for t in range(T):
        u_minus.append(np.asscalar(U[t].x[0,0]-np.linalg.norm(U[t].G[0,:],ord=1)))
        u_plus.append(np.asscalar(U[t].x[0,0]+np.linalg.norm(U[t].G[0,:],ord=1)))
    ax1.fill_between(range(T),u_minus,u_plus,color='purple',alpha=0.5)
    ax0.set_xlabel(r'time',fontsize=26)
    ax0.set_ylabel(r'$x_1$',fontsize=26)
    ax0.set_title(r'Performance Output Over Time',fontsize=26)
    ax1.set_title(r'Control Input Over Time',fontsize=26)
    ax1.set_xlabel(r'time',fontsize=26)
    ax1.set_ylabel(r'$u$',fontsize=26)
#    ax1.set_title(r'Possible Control Inputs Over Time',fontsize=26)
#    ax2.set_title(r'Error of Observer State')
    if "TI-LQG" in keys:
        L,K=LQG(S.A[0],S.B[0],S.C[0],S.W[0].G**2,S.V[0].G**2,S.QQ[0],S.RR[0])
    if "TV-LQG" in keys:
        S.L,S.K=LQG_LTV(S,T)
    for i in range(N):
        zeta_x=2*(np.random.random((S.n,1))-0.5)
        zeta_x=1*np.ones((S.n,1))*(-1)**np.random.randint(1,3)
        x_0=np.dot(S.X0.G,zeta_x)+S.X0.x
#        x_0=np.random.multivariate_normal(mean=S.X0.x.reshape(S.n),cov=S.X0.G).reshape(S.n,1)
        w,v=generate_random_disturbance(S,T,method=disturbance_method)
        x,y,u,x_observer={},{},{},{}
        for method in keys:
            if method=="Our Method":
                x[method],y[method],u[method],x_observer[method]=simulate_my_controller(S,x_0,T,w,v)
            elif method=="TI-LQG":
                x[method],y[method],u[method],x_observer[method]=simulate_LQG_LTI(S,x_0,T,w,v,L,K)
            elif method=="TV-LQG":
                x[method],y[method],u[method],x_observer[method]=simulate_LQG_LTV(S,x_0,T,w,v)
        # Plot
        c={"TI-LQG":"green","TV-LQG":"blue","Our Method":"Red"}
        for method in keys:
            ax0.plot(range(T+1),[np.asscalar(x[method][t][0,0]) for t in range(T+1)],'-',Linewidth=5,color=c[method],label=method)
            ax0.plot(range(T+1),[np.asscalar(x[method][t][0,0]) for t in range(T+1)],'o',Linewidth=5,color=c[method])
            ax0.plot(range(T+1),[0 for t in range(T+1)],'--',Linewidth=1,color="black")
            ax0.grid(lw=0.2,color=(0.2,0.3,0.2))
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend(handles,labels,fontsize=20)
        for method in keys:
            ax1.plot(range(T),[np.asscalar(u[method][t][0,0]) for t in range(T)],'-',Linewidth=5,color=c[method],label=method)
            ax1.plot(range(T),[np.asscalar(u[method][t][0,0]) for t in range(T)],'o',Linewidth=5,color=c[method])
            ax1.plot(range(T),[0 for t in range(T)],'--',Linewidth=1,color="black")
            ax1.grid(lw=0.2,color=(0.2,0.3,0.2))

        handles, labels = ax0.get_legend_handles_labels()
        ax1.legend(handles,labels,fontsize=20)
        J={}
        for method in x.keys():
            J[method]=sum([np.linalg.norm(np.dot(S.D[t],x[method][t]),ord=2)**2+np.linalg.norm(u[method][t],ord=2)**2 for t in range(T)])
            J[method]+=np.linalg.norm(np.dot(S.D[T],x[method][T]),ord=2)**2
        print(J)
    return J

J=simulate_and_plot(N=1,disturbance_method="extreme",keys=["Our Method","TV-LQG","TI-LQG"])

#N=1000
#J=simulate_and_cost_evaluate(N=N,disturbance_method="guassian",keys=["Our Method","TV-LQG","TI-LQG"])
#a=np.array([J[i]["Our Method"]/J[i]["TV-LQG"] for i in range(N)])
#b=np.array([J[i]["Our Method"]/J[i]["TI-LQG"] for i in range(N)])
#print(np.mean(a),np.mean(b))
