
import numpy as np
import random 
import matplotlib.pyplot as plt
import math


iter=3000
N=20
N_row=50
N_col=10
n=10

########################
#The local training period
tau=4
#tau=8
#tau=12
#tau=16


A=np.zeros((N,N_row,N_col))
b=np.zeros((N,N_row))
C=np.zeros((N,n,n))
D=np.zeros((N,n))
for s1 in range(0,N):
    for s2 in range(0,N_row):
        b[s1][s2]=random.uniform(0,1)
        for s3 in range(0,N_col):
            A[s1][s2][s3]=random.uniform(0,1)
            
            
        
     
W=np.ones((N,N))

W1=(1/N)*np.ones((N,N))

W2=np.zeros((N,N))
for i in range(0,N):
    W2[i][i]=1

    
L=np.zeros(N)
for i in range(0,N):
    L[i]=np.linalg.norm(np.dot(np.transpose(A[i]),A[i]))
    
L_max=L[0]
for i in range(0,N):
    if L[i]>L_max:
        L_max=L[i]


    
initial_counter_setting=8

Opt1=0*C[1]
Opt2=0*D[1]


for sss in range(0,N):
    C[sss]=np.dot(np.transpose(A[sss]),A[sss])
    Opt1=Opt1+C[sss]
    D[sss]=np.dot(np.transpose(A[sss]),b[sss])
    Opt2=Opt2+D[sss]
    
X_OPT=np.dot(np.linalg.inv(Opt1),Opt2)



def obj(XX):
    result=0
    for p in range(0,N):
        result=result+np.linalg.norm(np.dot(A[p],XX)-b[p])*np.linalg.norm(np.dot(A[p],XX)-b[p])
    final_result=result/(2*N)
    return final_result

optimal_value=obj(X_OPT)

######################################################################################
######################################################################################
#FedRecu

step_size=1/((13/8)*L_max*tau)  
Result_Recur=np.zeros(iter)
Result_Recur1=np.zeros(iter)

X=np.zeros((iter,N,n))

initial_value=0*X[0,i]
for i in range(0,N):
    initial_value=initial_value+X[0,i]-step_size*(np.dot(np.dot(np.transpose(A[i]),A[i]),X[0,i])-np.dot(np.transpose(A[i]),b[i]))

for i in range(0,N):
    X[1,i]=initial_value/N
    

counter=0


for t in range(0,iter-2):
    if (t+1)%tau==0:
        counter=counter+1
        for i in range(0,N):
            VALUE=0*X[t,i]
            for j in range(0,N):
                g1=(np.dot(np.dot(np.transpose(A[j]),A[j]),X[t+1,j])-np.dot(np.transpose(A[j]),b[j])) 
                g2=(np.dot(np.dot(np.transpose(A[j]),A[j]),X[t,j])-np.dot(np.transpose(A[j]),b[j]))
                VALUE=VALUE+(2*X[t+1,j]-X[t,j]-step_size*g1+step_size*g2)
            X[t+2,i]=VALUE/N 
            print(t+1)
            print(obj(X[t+2,i])-optimal_value)
        Result_Recur1[counter]=obj(X[t+2,0])-optimal_value
    elif (t)%tau==0:
        for i in range(0,N):
            VALUE=0*X[t,i] 
            for j in range(0,N):
                g1=(np.dot(np.dot(np.transpose(A[j]),A[j]),X[t+1,j])-np.dot(np.transpose(A[j]),b[j]))  
                g2=(np.dot(np.dot(np.transpose(A[j]),A[j]),X[t,j])-np.dot(np.transpose(A[j]),b[j]))
                VALUE=VALUE+(X[t,j]+step_size*g1-step_size*g2)
            X[t+2,i]=2*X[t+1,i]-VALUE/N 
    else:
        for i in range(0,N):
            g1=(np.dot(np.dot(np.transpose(A[i]),A[i]),X[t+1,i])-np.dot(np.transpose(A[i]),b[i])) 
            g2=(np.dot(np.dot(np.transpose(A[i]),A[i]),X[t,i])-np.dot(np.transpose(A[i]),b[i]))
            X[t+2,i]=2*X[t+1,i]-X[t,i]-step_size*g1+step_size*g2


for t in range(0,iter-2-initial_counter_setting):
    Result_Recur[t]=Result_Recur1[t+initial_counter_setting]



######################################################################################
######################################################################################
#scaffold

Result_SCAFFOLD=np.zeros(iter)

X_my2=0*np.zeros((N,n))
Y2=0*np.zeros((N,n))  
X2=0*np.zeros(n)  
step_size2g=1
step_size2l=1/(81*L_max*tau)
C2=0*np.zeros((N,n))
C_PLUS=0*np.zeros((N,n))
C_ALL=0*np.zeros(n)  

Delta_Y2=0*np.zeros((N,n))  
Delta_C2=0*np.zeros((N,n))



Result_SCAFFOLD[0]=obj(X[initial_counter_setting*tau+1,0])-optimal_value

X2=X[initial_counter_setting*tau+1,0]

for t in range(1,iter):
    
    for i in range(0,N):
        Y2[i]=X2
        for k in range(0,tau):
            Y2[i]=Y2[i]-step_size2l*(((np.dot(np.dot(np.transpose(A[i]),A[i]),Y2[i])-np.dot(np.transpose(A[i]),b[i])))-C2[i]+C_ALL)
        C_PLUS[i]=(np.dot(np.dot(np.transpose(A[i]),A[i]),X2)-np.dot(np.transpose(A[i]),b[i]))
        Delta_Y2[i]=Y2[i]-X2
        Delta_C2[i]=C_PLUS[i]-C2[i]
        C2[i]=C_PLUS[i]
    AAA=0*Delta_Y2[0]
    BBB=0*Delta_C2[0]
    for j in range(0,N):
        AAA=AAA+(1/N)*Delta_Y2[j]
        BBB=BBB+(1/N)*Delta_C2[j]
    X2=X2+step_size2g*AAA
    C_ALL=C_ALL+BBB
    
    print(t)
    print(obj(X2)-optimal_value)
    Result_SCAFFOLD[t]=obj(X2)-optimal_value



######################################################################################
######################################################################################
#FedTrack and FedLin


step_size2=1/(10*L_max*tau)

step_size3=1/(18*L_max*tau)


Result_my1=np.zeros(iter)
Result_my2=np.zeros(iter)
Result_my3=np.zeros(iter)
Result_my4=np.zeros(iter)


GG_my2=np.zeros(n)
GG_AVG_my2=np.zeros(n)
ave_X_my2=X[initial_counter_setting*tau+1,0]

GG_my3=np.zeros(n)
GG_AVG_my3=np.zeros(n)
ave_X_my3=X[initial_counter_setting*tau+1,0]



X_my2=np.zeros((N,n))
X_my3=np.zeros((N,n))


Result_my2[0]=obj(X[initial_counter_setting*tau+1,0])-optimal_value
Result_my3[0]=obj(X[initial_counter_setting*tau+1,0])-optimal_value


for t in range(1,iter):
    
    
    
    
    for i in range(0,N):   
        X_my2[i]=ave_X_my2
        X_my3[i]=ave_X_my3
        
        GG_FITST_my2=np.dot(np.dot(np.transpose(A[i]),A[i]),X_my2[i])-np.dot(np.transpose(A[i]),b[i])
        GG_FITST_my3=np.dot(np.dot(np.transpose(A[i]),A[i]),X_my3[i])-np.dot(np.transpose(A[i]),b[i])
        
        for k in range(0,tau):
            X_my2[i]=X_my2[i]-step_size2*(GG_AVG_my2-GG_FITST_my2+np.dot(np.dot(np.transpose(A[i]),A[i]),X_my2[i])-np.dot(np.transpose(A[i]),b[i]))
            X_my3[i]=X_my3[i]-step_size3*(GG_AVG_my3-GG_FITST_my3+np.dot(np.dot(np.transpose(A[i]),A[i]),X_my3[i])-np.dot(np.transpose(A[i]),b[i]))
            
            
    ave_X_my2=np.zeros(n)
    GG_my2=np.zeros(n)
    ave_X_my3=np.zeros(n)
    GG_my3=np.zeros(n)
    
    for j in range(0,N):
        ave_X_my2=ave_X_my2+X_my2[j]
        ave_X_my3=ave_X_my3+X_my3[j]
        
    ave_X_my2=ave_X_my2/N
    ave_X_my3=ave_X_my3/N
    
    for ii in range(0,N):       
        GG_my2=GG_my2+np.dot(np.dot(np.transpose(A[ii]),A[ii]),ave_X_my2)-np.dot(np.transpose(A[ii]),b[ii])
        GG_my3=GG_my3+np.dot(np.dot(np.transpose(A[ii]),A[ii]),ave_X_my3)-np.dot(np.transpose(A[ii]),b[ii])
        
    GG_AVG_my2=GG_my2/N
    GG_AVG_my3=GG_my3/N
    
    
    print(t)
    print(obj(ave_X_my2)-obj(X_OPT))
    print(obj(ave_X_my3)-obj(X_OPT))
    
    Result_my2[t]=obj(ave_X_my2)-obj(X_OPT)
    Result_my3[t]=obj(ave_X_my3)-obj(X_OPT)




######################################################################################
######################################################################################
#tau=4
np.savetxt('Result_FedRecu_TAU4.txt',Result_Recur,fmt="%f",delimiter=" ")     
np.savetxt('Result_SCAFFOLD_TAU4.txt',Result_SCAFFOLD,fmt="%f",delimiter=" ")
np.savetxt('Result_FedLin_TAU4.txt',Result_my2,fmt="%f",delimiter=" ")
np.savetxt('Result_FedTrack_TAU4.txt',Result_my3,fmt="%f",delimiter=" ")

######################################################################################
######################################################################################
#tau=8
#np.savetxt('Result_FedRecu_TAU8.txt',Result_Recur,fmt="%f",delimiter=" ")     
#np.savetxt('Result_SCAFFOLD_TAU8.txt',Result_SCAFFOLD,fmt="%f",delimiter=" ")
#np.savetxt('Result_FedLin_TAU8.txt',Result_my2,fmt="%f",delimiter=" ")
#np.savetxt('Result_FedTrack_TAU8.txt',Result_my3,fmt="%f",delimiter=" ")


######################################################################################
######################################################################################
#tau=12
#np.savetxt('Result_FedRecu_TAU12.txt',Result_Recur,fmt="%f",delimiter=" ")     
#np.savetxt('Result_SCAFFOLD_TAU12.txt',Result_SCAFFOLD,fmt="%f",delimiter=" ")
#np.savetxt('Result_FedLin_TAU12.txt',Result_my2,fmt="%f",delimiter=" ")
#np.savetxt('Result_FedTrack_TAU12.txt',Result_my3,fmt="%f",delimiter=" ")

######################################################################################
######################################################################################
#tau=16
#np.savetxt('Result_FedRecu_TAU16.txt',Result_Recur,fmt="%f",delimiter=" ")     
#np.savetxt('Result_SCAFFOLD_TAU16.txt',Result_SCAFFOLD,fmt="%f",delimiter=" ")
#np.savetxt('Result_FedLin_TAU16.txt',Result_my2,fmt="%f",delimiter=" ")
#np.savetxt('Result_FedTrack_TAU16.txt',Result_my3,fmt="%f",delimiter=" ")