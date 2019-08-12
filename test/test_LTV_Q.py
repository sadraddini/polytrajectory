#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:52:30 2019

@author: sadra
"""

import numpy as np
from pypolytrajectory.LTV import system,test_controllability
from pypolytrajectory.reduction import reduced_order,order_reduction_error,error_construction,error_construction_old
from pypolycontain.lib.objects import zonotope
from pypolycontain.lib.zonotope_order_reduction.methods import G_cut,Girard
from pypolytrajectory.synthesis import synthesis_disturbance_feedback,zonotopic_controller,synthesis,zonotopic_controller_soft,\
    output_feedback_synthesis,outputfeedback_synthesis_zonotope_solution,output_feedback_synthesis_lightweight,\
    triangular_stack_list,output_feedback_synthesis_lightweight_many_variables
from pypolytrajectory.system import LQG_LTV,LTV,LQG
import scipy.linalg as spa


np.random.seed(10)
 
S=LTV()
n=20
m=1
o=1
z=2
T=58
S.X0=zonotope(np.ones((n,1))*0,np.eye(n)*1)
B=np.random.randint(0,2,size=(n,m))
B[0,0]=1
B[1,0]=0
A=0.8*np.eye(n)+np.random.randint(-100,100,size=(n,n))*0.01*0.1
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
    S.V[t]=zonotope(np.zeros((o,1)),np.eye(o)*0.01)
    S.QQ[t]=np.eye(n)*1
    S.RR[t]=np.eye(m)*1
S.F_cost=np.eye(n)*0.00

S.U_set=zonotope(np.zeros((m,1)),np.eye(m)*1)
S.construct_dimensions()
#S.construct_E()
S._abstract_evolution_matrices()
S._error_zonotopes()



#T=10
#import scipy.linalg as spa
#prog=MP.MathematicalProgram()
#sys=S
#theta,P,Q,alpha,beta,gamma={},{},{},{},{},{}
## Main variables
#for t in range(T):
#    theta[t]=prog.NewContinuousVariables(sys.m,sys.o*(t+1),"theta%d"%t)
#VarTheta=np.hstack([theta[t] for t in range(T)])
#Theta_T=triangular_stack_list([theta[t] for t in range(T)])
#for t in range(T):
##    I_lar=spa.block_diag(*[np.eye(sys.o*(tau+1)) for tau in range(t+1)])
##    vu[t]=np.vstack(( I_lar, np.zeros((VarTheta.shape[1]-I_lar.shape[1],I_lar.shape[1])) ))
##    mu[t]=triangular_stack_list([np.eye(sys.o*(tau+1)) for tau in range(t+1)])
#    P[t]=np.hstack(( np.eye(sys.m*(t+1)),np.zeros((sys.m*(t+1),sys.m*(T-t-1))) ))
#    Q[t]=np.vstack(( np.eye(sys.o*(t+1)),np.zeros((sys.o*(T-t-1) , sys.o*(t+1) )) ))
#alpha[0]=np.eye(sys.o)
#beta[0]=np.zeros((sys.o*(t+1),Theta_T.shape[0]))
#gamma[0]=np.zeros((Theta_T.shape[1],sys.o*(t+1)))
#for t in range(T):
#    bigMatrix1=np.hstack(( np.eye(sys.o*(t+1)) ,  np.zeros((sys.o*(t+1),sys.m*(t+1)))   ))
#    bigMatrix2=np.hstack(( sys.M[t] , sys.N[t]  ))
#    print bigMatrix1.shape,bigMatrix2.shape
#    bigMatrix=np.vstack(( bigMatrix1 ,  bigMatrix2   ))
#    # alpha
#    alpha_0=np.vstack(( alpha[t],np.zeros(( sys.m*(t+1),sys.o*(t+1) )) ))
#    Io=np.hstack(( np.eye(sys.o*(t+1)) , np.zeros(( sys.o*(t+1),sys.o )) ))
#    I1=spa.block_diag(*[0*np.eye( sys.o*(t+1) ) , np.eye(sys.o)])
#    print bigMatrix.shape, alpha_0.shape, Io.shape,I1.shape
#    alpha[t+1]=np.dot(bigMatrix,alpha_0).dot(Io)+I1
#    # beta
#    betaP=np.vstack(( beta[t] , P[t] ))
#    beta[t+1]=np.dot(bigMatrix, betaP )
#    
#if True:
#    raise NotImplementedError

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


#t=7
#a1=np.dot(S.D[t],S.P["x",t])-np.dot(S.R[t],S.Q["x",t])
#print np.linalg.norm(a1,"fro") 
#print np.linalg.norm(S.F[t].G[n:,:],"fro") 
#print np.linalg.norm(S.F[t].G[:n,:],"fro") 
#R=np.dot(np.dot(S.D[t],S.P["x",t]),np.linalg.pinv(S.Q["x",t]))
#print np.linalg.norm(S.R[t]-R,"fro") 
#a2=np.dot(S.D[t],S.P["x",t])-np.dot(R,S.Q["x",t])
#a3=np.dot(S.D[t],S.P["x",t])-np.linalg.multi_dot([S.D[t],S.P["x",t],np.linalg.pinv(S.Q["x",t]),S.Q["x",t]])
#a4=np.dot(S.D[t],S.P["x",t])-np.linalg.multi_dot([S.D[t],S.P["x",t],np.eye(n)])
#
#print "a2",np.linalg.norm(a2,"fro") 
#print "a3",np.linalg.norm(a3,"fro") 
#print "a4",np.linalg.norm(a3,"fro") 
#
#eye=np.dot(S.Q["x",t],np.linalg.pinv(S.Q["x",t]))
#print np.linalg.norm(eye-np.eye(t+1),"fro") 
#eye2=np.dot(np.linalg.pinv(S.Q["x",t]),S.Q["x",t])
#print np.linalg.norm(eye2-np.eye(n),"fro") 


T=40
#pi,J=output_feedback_synthesis_lightweight(S,T=T)
#a=J.Jacobian(pi[3])
#raise 1
u_tilde,theta=output_feedback_synthesis_lightweight_many_variables(S,T=T)
Z,U=outputfeedback_synthesis_zonotope_solution(S,u_tilde,theta)


# Synthesis
Goal=zonotope(np.ones((1,1))*0,np.eye(1)*1)

S.L,S.K=LQG_LTV(S,T)


#if False:
#    for t in range(T+2):
#        print t,"-zonotope reduction"
#        G=Girard(Z[t].G,1)
#        S.Z[t].G=G
#        print G
#    synthesis(S,T=T,y_goal=Goal)
#else:
#    error_construction(S,T+1,q0=1)
#    order_reduction_error(S,T+1,q=2)
#    synthesis_disturbance_feedback(S,T=T,y_goal=Goal,control_bound=False)

    
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
            
def simulate_LQG_LTI(sys,x_0,T,w,v):
    L,K=LQG(sys.A[0],sys.B[0],sys.C[0],sys.W[0].G,sys.V[0].G,sys.QQ[0],sys.RR[0])
    x,y,u,x_observer={},{},{},{}
    x[0]=x_0
    y[0]=np.dot(sys.C[0],x[0])+v[0]    
#    x_observer[0]=np.dot(np.linalg.pinv(S.C[0]),y[0])
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
        x["Our Method"],y["Our Method"],u["Our Method"],x_observer["Our Method"]=simulate_my_controller(S,x_0,T,w,v)
        for method in x.keys():
            J[i][method]=sum([np.asscalar(y[method][t])**2+np.asscalar(u[method][t])**2 for t in range(T)])
    return J
        
        
def simulate_and_plot(N=1,disturbance_method="extreme"):
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
    ax0.fill_between(range(T+1),y_minus,y_plus,alpha=0.5,color='red')
    y_minus,y_plus,u_minus,u_plus=[],[],[],[]
    for t in range(T):
        u_minus.append(np.asscalar(U[t].x[0,0]-np.linalg.norm(U[t].G[0,:],ord=1)))
        u_plus.append(np.asscalar(U[t].x[0,0]+np.linalg.norm(U[t].G[0,:],ord=1)))
    ax1.fill_between(range(T),u_minus,u_plus,color='green',alpha=0.5)
    ax0.set_xlabel(r'time',fontsize=26)
    ax0.set_ylabel(r'$x_1$',fontsize=26)
    ax0.set_title(r'Reachable Outputs Over Time',fontsize=26)
    ax1.set_xlabel(r'time',fontsize=26)
    ax1.set_ylabel(r'$u$',fontsize=26)
#    ax1.set_title(r'Possible Control Inputs Over Time',fontsize=26)
#    ax2.set_title(r'Error of Observer State')
    for i in range(N):
        zeta_x=2*(np.random.random((S.n,1))-0.5)
        zeta_x=1*np.ones((S.n,1))*(-1)**np.random.randint(1,3)
        x_0=np.dot(S.X0.G,zeta_x)+S.X0.x
        w,v=generate_random_disturbance(S,T,method=disturbance_method)
        x,y,u,x_observer={},{},{},{}
        x["TI-LQG"],y["TI-LQG"],u["TI-LQG"],x_observer["TI-LQG"]=simulate_LQG_LTI(S,x_0,T,w,v)
        x["TV-LQG"],y["TV-LQG"],u["TV-LQG"],x_observer["TV-LQG"]=simulate_LQG_LTV(S,x_0,T,w,v)
        x["Our Method"],y["Our Method"],u["Our Method"],x_observer["Our Method"]=simulate_my_controller(S,x_0,T,w,v)
        # Plot
        c={"TI-LQG":"green","TV-LQG":"blue","Our Method":"Red"}
        keys=["Our Method","TV-LQG","TI-LQG"]
        for method in keys:
            ax0.plot(range(T+1),[np.asscalar(x[method][t][0,0]) for t in range(T+1)],'-',Linewidth=5,color=c[method],label=method)
            ax0.plot(range(T+1),[np.asscalar(x[method][t][0,0]) for t in range(T+1)],'o',Linewidth=5,color=c[method])
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend(handles,labels,fontsize=20)
        for method in keys:
            ax1.plot(range(T),[np.asscalar(u[method][t][0,0]) for t in range(T)],'-',Linewidth=5,color=c[method],label=method)
            ax1.plot(range(T),[np.asscalar(u[method][t][0,0]) for t in range(T)],'o',Linewidth=5,color=c[method])
        handles, labels = ax0.get_legend_handles_labels()
        ax1.legend(handles,labels,fontsize=20)
        J={}
        for method in x.keys():
            J[method]=sum([np.linalg.norm(x[method][t],ord=2)**2+np.linalg.norm(u[method][t],ord=2)**2 for t in range(T)])
        print J
    return J

J=simulate_and_plot(N=1,disturbance_method="extreme")
#J={}
#N=20
#J["extreme"]=simulate_and_cost_evaluate(N,disturbance_method="extreme")
#J["guassian"]=simulate_and_cost_evaluate(N,disturbance_method="guassian")
#
#fig_J, axJ = plt.subplots()
#fig_C, axC = plt.subplots()
##plt_J.hist([J["guassian"][i]["TV-LQG"] for i in range(N)],color="blue",label="TV-LQG")
###plt_J.hist([J["extreme"][i]["TI-LQG"] for i in range(N)],color="green",label="TI-LQG")
##plt_J.hist([J["guassian"][i]["Our Method"] for i in range(N)],color="red",label="Our Method")
#ratio_guassian=[J["guassian"][i]["Our Method"]/J["guassian"][i]["TV-LQG"] for i in range(N)]
#ratio_extreme=[J["extreme"][i]["Our Method"]/J["extreme"][i]["TV-LQG"] for i in range(N)]
#axJ.hist(ratio_guassian,color="red")
#axC.hist(ratio_extreme,color="blue")
#print "Guassian Ratio:",np.mean(np.array(ratio_guassian)),"variance:",np.var(np.array(ratio_guassian))
#print "Extreme Ratio:",np.mean(np.array(ratio_extreme)),"variance:",np.var(np.array(ratio_extreme))