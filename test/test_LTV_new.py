#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 18:53:03 2019

@author: sadra
"""

import numpy as np
from pypolytrajectory.LTV import system,test_controllability
from pypolytrajectory.reduction import reduced_order,order_reduction_error,error_construction,error_construction_old
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.zonotope_order_reduction.methods import G_cut,Girard
from pypolytrajectory.synthesis import synthesis_disturbance_feedback,zonotopic_controller,synthesis,zonotopic_controller_soft
from pypolytrajectory.system import LQG_LTV,LTV,LQG


 
S=LTV()
n=5
m=1
o=1
T=58
S.X0=zonotope(np.ones((n,1))*0,np.eye(n)*1)
B=np.random.randint(0,2,size=(n,m))
B[0,0]=1
B[1,0]=0
A=0.0*np.eye(n)+np.random.normal(size=(n,n))*0.6
C=np.zeros((o,n))
C[0,0]=1
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
    S.D[t]=np.eye(n)
    S.W[t]=zonotope(np.zeros((n,1)),np.eye(n)*0.01)
    S.V[t]=zonotope(np.zeros((o,1)),np.eye(o)*0.00001)
    S.Q[t]=np.eye(n,n)*1
    S.R[t]=np.eye(m)*1
S.F_cost=np.eye(n)*0.00

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

S.U_set=zonotope(np.zeros((m,1)),np.eye(m)*1)
S.construct_dimensions()
S.construct_E()
M,N,Z=reduced_order(S,T-1)

S.Z=Z
S.M=M
S.N=N

import matplotlib.pyplot as plt0
plt0.plot(range(T-1),[np.linalg.norm(Z[t].G[0,:],1) for t in range(T-1)],LineWidth=2,color='green')
plt0.plot(range(T-1),[np.linalg.norm(Z[t].G[0,:],1) for t in range(T-1)],'o',MarkerSize=3,color='black')
plt0.title(r"Error Over Time")

# Synthesis
Goal=zonotope(np.ones((1,1))*0,np.eye(1)*1)

T=35
S.L,S.K=LQG_LTV(S,T)
S.L["LTI"],S.K["LTI"]=LQG(S.A[0],S.B[0],S.C[0],S.W[0].G**2,S.V[0].G**2,S.Q[0],S.R[0])


if False:
    for t in range(T+2):
        print t,"-zonotope reduction"
        G=Girard(Z[t].G,1)
        S.Z[t].G=G
        print G
    synthesis(S,T=T,y_goal=Goal)
else:
    error_construction(S,T+1,q0=1)
    order_reduction_error(S,T+1,q=2)
    synthesis_disturbance_feedback(S,T=T,y_goal=Goal,control_bound=False)

    
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


def simulate_LQG_LTV(sys,x_0,T,w,v):
    x,y,u,x_observer={},{},{},{}
    x[0]=x_0
    y[0]=np.dot(sys.C[0],x[0])+v[0]    
    x_observer[0]=np.dot(np.linalg.pinv(S.C[0]),y[0])
    x_observer[0]=x_0
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
            
def simulate_LQG_LTI(sys,x_0,T,w,v):
    x,y,u,x_observer={},{},{},{}
    x[0]=x_0
    y[0]=np.dot(sys.C[0],x[0])+v[0]    
#    x_observer[0]=np.dot(np.linalg.pinv(S.C[0]),y[0])
    x_observer[0]=x_0
    x_observer[0]=sys.X0.x
    for t in range(T+1):
#        print "simulating observer time:",t
        if t==T:
            return x,y,u,x_observer
        u[t]=np.dot(-S.K["LTI"],x_observer[t])
        x[t+1]=np.dot(sys.A[t],x[t])+np.dot(sys.B[t],u[t])+w[t]
        y[t+1]=np.dot(sys.C[t+1],x[t+1])+v[t+1]
        x_observer[t+1]=np.dot(sys.A[t],x_observer[t])+np.dot(sys.B[t],u[t])+\
            np.dot(S.L["LTI"],y[t+1]-np.dot(sys.C[t+1],np.dot(sys.A[t],x_observer[t])+np.dot(sys.B[t],u[t])))
#        print "control",t,u[t],u[t].shape

def simulate_zonotope(sys,x_0,T,w,v):
    x,y,u={},{},{}
    x[0]=x_0
    for t in range(T+1):
        print "simulating time:",t
        y[t]=np.dot(sys.C[t],x[t])+v[t]
        if t==T:
            return x,y,u,x
        Y=np.vstack([y[tau] for tau in range(t+1)])
        Ybar=np.vstack([sys.ybar[tau] for tau in range(t+1)])
        if t<T+2:
            zono_Y=zonotope(Ybar,np.dot(sys.Phi[t],sys.E[t-1].G))
            zono_U=zonotope(sys.ubar[t],np.dot(sys.theta[t],sys.E[t-1].G))
            u[t]=zonotopic_controller(Y,zono_Y,zono_U)
        else:
            Ubar=np.vstack([sys.ubar[tau] for tau in range(t)])
            U=np.vstack([u[tau] for tau in range(t)])
            YU=np.vstack((Y,U))
            YU_bar=np.vstack((Ybar,Ubar))
            phi_E=np.dot(sys.Phi[t],sys.E[t-1].G)
            theta_E=np.dot(sys.Theta[t-1],sys.E[t-2].G)
            theta_E_0=np.hstack((theta_E,np.zeros((sys.Theta[t-1].shape[0],phi_E.shape[1]-theta_E.shape[1]))))
            G=np.vstack((phi_E,theta_E_0))
            zono_Y=zonotope(YU_bar,G)
            zono_U=zonotope(sys.ubar[t],np.dot(sys.theta[t],sys.E[t-1].G))
            K=np.dot(np.dot(sys.theta[t],sys.E[t-1].G),np.linalg.pinv(G))
            u[t]=sys.ubar[t]+np.dot(K,YU-YU_bar)
            u[t]=zonotopic_controller(YU,zono_Y,zono_U)
        x[t+1]=np.dot(sys.A[t],x[t])+np.dot(sys.B[t],u[t])+w[t]  
        
    
def simulate_and_cost_evaluate(N=1,disturbance_method="extreme"):
    J={}
    for i in range(N):
        print "iteration",i
        J[i]={}
        zeta_x=2*(np.random.random((S.n,1))-0.5)
        x_0=np.dot(S.X0.G,zeta_x)+S.X0.x
        w,v=generate_random_disturbance(S,T,method=disturbance_method)
        x,y,u,x_observer={},{},{},{}
        x["TI-LQG"],y["TI-LQG"],u["TI-LQG"],x_observer["TI-LQG"]=simulate_LQG_LTI(S,x_0,T,w,v)
        x["TV-LQG"],y["TV-LQG"],u["TV-LQG"],x_observer["TV-LQG"]=simulate_LQG_LTV(S,x_0,T,w,v)
        x["Zonotope"],y["Zonotope"],u["Zonotope"],x_observer["Zonotope"]=simulate_zonotope(S,x_0,T,w,v)
        for method in x.keys():
            J[i][method]=sum([np.asscalar(y[method][t])**2+np.asscalar(u[method][t])**2 for t in range(T)])
    return J
        
        
def simulate_and_plot(N=1,disturbance_method="extreme"):
    import matplotlib.pyplot as plt
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig0.set_size_inches(12, 8)
    fig1.set_size_inches(12, 8)
    y_minus,y_plus,u_minus,u_plus=[],[],[],[]
    for t in range(T+1):
        y_minus.append(np.asscalar(S.ybar[t]-np.linalg.norm(S.phi_E[t],ord=np.inf)))
        y_plus.append(np.asscalar(S.ybar[t]+np.linalg.norm(S.phi_E[t],ord=np.inf)))
    ax0.fill_between(range(T+1),y_minus,y_plus,alpha=0.5,color='red')
    y_minus,y_plus,u_minus,u_plus=[],[],[],[]
    for t in range(T):
        u_minus.append(np.asscalar(S.ubar[t]-np.linalg.norm(S.theta_E[t],ord=np.inf)))
        u_plus.append(np.asscalar(S.ubar[t]+np.linalg.norm(S.theta_E[t],ord=np.inf)))
    ax1.fill_between(range(T),u_minus,u_plus,color='green',alpha=0.5)
    ax0.set_xlabel(r'time',fontsize=26)
    ax0.set_ylabel(r'$y$',fontsize=26)
    ax0.set_title(r'Reachable Outputs Over Time',fontsize=26)
    ax1.set_xlabel(r'time',fontsize=26)
    ax1.set_ylabel(r'$u$',fontsize=26)
    ax2.set_title(r'Error of Observer States TV-LQG')
    ax3.set_title(r'Error of Observer States TI-LQG')
    for i in range(N):
        zeta_x=2*(np.random.random((S.n,1))-0.5)
        x_0=np.dot(S.X0.G,zeta_x)+S.X0.x
        w,v=generate_random_disturbance(S,T,method=disturbance_method)
        x,y,u,x_observer={},{},{},{}
        x["TI-LQG"],y["TI-LQG"],u["TI-LQG"],x_observer["TI-LQG"]=simulate_LQG_LTI(S,x_0,T,w,v)
        x["TV-LQG"],y["TV-LQG"],u["TV-LQG"],x_observer["TV-LQG"]=simulate_LQG_LTV(S,x_0,T,w,v)
        try:
            x["Zonotope"],y["Zonotope"],u["Zonotope"],x_observer["Zonotope"]=simulate_zonotope(S,x_0,T,w,v)
        except:
            print "Zonotope Method Failed"
            x["Zonotope"],y["Zonotope"],u["Zonotope"],x_observer["Zonotope"]=x["TI-LQG"],y["TI-LQG"],u["TI-LQG"],x_observer["TI-LQG"]
        # Plot
        c={"TI-LQG":"green","TV-LQG":"blue","Zonotope":"Red"}
        keys=["Zonotope","TV-LQG"]
        for method in keys:
            ax0.plot(range(T+1),[np.asscalar(y[method][t]) for t in range(T+1)],'-',Linewidth=5,color=c[method],label=method)
            ax0.plot(range(T+1),[np.asscalar(y[method][t]) for t in range(T+1)],'.',Linewidth=5,color=c[method])
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend(handles,labels,fontsize=20)
        for method in keys:
            ax1.plot(range(T),[np.asscalar(u[method][t]) for t in range(T)],'-',Linewidth=5,color=c[method],label=method)
            ax1.plot(range(T),[np.asscalar(u[method][t]) for t in range(T)],'.',Linewidth=5,color=c[method])
        handles, labels = ax0.get_legend_handles_labels()
        ax1.legend(handles,labels,fontsize=20)
        for i in range(S.n):
            ax2.plot(range(T+1),[np.asscalar(x_observer["TV-LQG"][t][i,0]-x["TV-LQG"][t][i,0])\
                     for t in range(T+1)],'-',Linewidth=1,color=(i/(S.n+0.1),1.0-i/(S.n+0.1),0.5))
            ax3.plot(range(T+1),[np.asscalar(x_observer["TI-LQG"][t][i,0]-x["TI-LQG"][t][i,0])\
                     for t in range(T+1)],'-',Linewidth=1,color=(i/(S.n+0.1),1.0-i/(S.n+0.1),0.5))
        J={}
        for method in x.keys():
            J[method]=sum([np.asscalar(y[method][t])**2+np.asscalar(u[method][t])**2 for t in range(T)])
        print J
        J_x={}
        for method in x.keys():
            J_x[method]=sum([np.linalg.norm(x[method][t])**2+np.asscalar(u[method][t])**2 for t in range(T)])
        print J_x
    return J

J=simulate_and_plot(N=1,disturbance_method="guassian")
#J={}
#N=20
#J["extreme"]=simulate_and_cost_evaluate(N,disturbance_method="extreme")
#J["guassian"]=simulate_and_cost_evaluate(N,disturbance_method="guassian")
#
#fig_J, axJ = plt.subplots()
#fig_C, axC = plt.subplots()
##plt_J.hist([J["guassian"][i]["TV-LQG"] for i in range(N)],color="blue",label="TV-LQG")
###plt_J.hist([J["extreme"][i]["TI-LQG"] for i in range(N)],color="green",label="TI-LQG")
##plt_J.hist([J["guassian"][i]["Zonotope"] for i in range(N)],color="red",label="Zonotope")
#ratio_guassian=[J["guassian"][i]["Zonotope"]/J["guassian"][i]["TV-LQG"] for i in range(N)]
#ratio_extreme=[J["extreme"][i]["Zonotope"]/J["extreme"][i]["TV-LQG"] for i in range(N)]
#axJ.hist(ratio_guassian,color="red")
#axC.hist(ratio_extreme,color="blue")
#print "Guassian Ratio:",np.mean(np.array(ratio_guassian)),"variance:",np.var(np.array(ratio_guassian))
#print "Extreme Ratio:",np.mean(np.array(ratio_extreme)),"variance:",np.var(np.array(ratio_extreme))