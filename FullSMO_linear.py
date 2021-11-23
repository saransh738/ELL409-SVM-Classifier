#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import random
import time
import math
from timeit import default_timer as timer

#Reading files
data_points_train = pd.read_csv('2019MT60763.csv', header = None, nrows = 3000)
data = np.array((data_points_train.sort_values(data_points_train.columns[25])).values)
dp = np.array(data)
text_x = (dp[:631,:25])
text_t = dp[:631,25].astype('int')

for i in range(text_t.shape[0]):
    if (text_t[i] == 0) :
        text_t[i] = 1
    else :
        text_t[i] = -1
        

tp_x_1 = np.append(dp[:100,:25],dp[306:406,:25],axis=0)
tp_t_1 = np.append(dp[:100,25],dp[306:406,25],axis=0)
tp_t_1 = tp_t_1.astype('int')


for i in range(tp_t_1.shape[0]):
    if (tp_t_1[i] == 0) :
        tp_t_1[i] = 1
    else :
        tp_t_1[i] = -1

tp_x_2 = np.append(dp[101:201,:25],dp[407:507,:25],axis=0)
tp_t_2 = np.append(dp[101:201,25],dp[407:507,25],axis=0)
tp_t_2 = tp_t_2.astype('int')

for i in range(tp_t_2.shape[0]):
    if (tp_t_2[i] == 0) :
        tp_t_2[i] = 1
    else :
        tp_t_2[i] = -1

tp_x_3 = np.append(dp[202:305,:25],dp[508:631,:25],axis=0)
tp_t_3 = np.append(dp[202:305,25],dp[508:631,25],axis=0)
tp_t_3 = tp_t_3.astype('int')

for i in range(tp_t_3.shape[0]):
    if (tp_t_3[i] == 0) :
        tp_t_3[i] = 1
    else :
        tp_t_3[i] = -1
        
def computing_L(alpha_i,alpha_j,C,Yi,Yj):
    if(Yi!=Yj):
        return max(0,alpha_j-alpha_i)
    else:
        return max(0,alpha_i+alpha_j-C)
    
def computing_H(alpha_i,alpha_j,C,Yi,Yj):
    if(Yi!=Yj):
        return min(C,C+alpha_j-alpha_i)
    else:
        return min(C,alpha_j+alpha_i)   

def computing_neta(Xi,Xj):
    return 2*(np.inner(Xi,Xj))-(np.inner(Xi,Xi))-(np.inner(Xj,Xj))

def computing_b2(b,Xi,Xj,yi,yj,alpha_j,alpha_i,alphaold_i,alphaold_j,Ej): 
    return b + Ej - yi * (alpha_i - alphaold_i) * np.inner(Xi, Xj) + yj * (alpha_j - alphaold_j) * np.inner(Xj, Xj) 

def computing_b1(b,Xi,Xj,yi,yj,alpha_j,alpha_i,alphaold_i,alphaold_j,Ei): 
    return b + Ei - yi * (alpha_i - alphaold_i) * np.inner(Xi, Xi) + yj * (alpha_j - alphaold_j) * np.inner(Xi, Xj) 

def computing_b(C,alpha_j,alpha_i,b1,b2):    
    if(alpha_i<C and alpha_i>0):
        return b1
    elif(alpha_j<C and alpha_j>0):
        return b2
    else:
        return (b1+b2)/2
 
 # calculating objective function 
def OF(alpha,y,x):
    s1 = np.sum(alpha)
    p = ((np.matmul(y.reshape((-1,1)) , y.reshape((1,-1)))) * (x.dot(x.T)) * (np.matmul(alpha.reshape((-1,1)),alpha.reshape((1,-1)))))
    s2 = np.sum(p)
    return s1 - (s2/2)

def Error_function(alpha,y,x,b):
    return np.dot((alpha * y), x.dot(x.T)) - b -y
    
def find_a2(alpha,y,Ei,Ej,neta,L,H):
    c = alpha - y*(Ei-Ej)/neta
    if(c<L):
        return L
    elif(c>H):
        return H
    else:
        return c

def find_aa2(L,H,Lobj,Hobj,eps,alph2):
    if (Lobj < Hobj-eps):
        return L
    elif (Lobj > Hobj+eps):
        return H
    else:
        return alph2
    
    
def TS(i1,i2,alpha,x,y,C,eps,b,m):
    if(i1 == i2):
        return 0
    alph1 = alpha[i1]
    alph2 = alpha[i2]
    E1 =  Error_function(alpha,y,x,b)[i1] 
    E2 =  Error_function(alpha,y,x,b)[i2] 
    s = y[i1]*y[i2]
    L =  computing_L(alph1,alph2,C,y[i1],y[i2])
    H =  computing_H(alph1,alph2,C,y[i1],y[i2])
    if(L==H):
        return 0
    neta = computing_neta(x[i1],x[i2])    
    if(neta>0):
        a2 = find_a2(alph2,y[i2],E1,E2,neta,L,H)
    else:
        A = alpha.copy()
        A[i2] = L
        Lobj = OF(A, y,x)
        A[i2] = H
        Hobj = OF(A, y,x)
        a2=find_aa2(L,H,Lobj,Hobj,eps,alph2)  
    if (a2 < 1e-4):
        a2 = 0.0
    elif (1e-4 > (C - a2)):
        a2 = C
        
    if (np.abs(a2-alph2) < eps*(a2+alph2+eps)):
        return 0
    a1 = alph1+s*(alph2-a2)
    b1 = computing_b1(b,x[i1],x[i2],y[i1],y[i2],a2,a1,alph1,alph2,E1)
    b2 = computing_b2(b,x[i1],x[i2],y[i1],y[i2],a2,a1,alph1,alph2,E2)
    b_new = computing_b(C,a2,a1,b1,b2)
    alpha[i1] = a1
    alpha[i2] = a2
    
    for i in range(i1,i2):
        for alph in range(int(a1),int(a2)):
            if(0.0<int(alph)<C):
                Error_function(alpha,y,x,b)[i] = 0.0
                
    for p in range(m):
        if(p!=i1 & p!=i2):
            Error_function(alpha,y,x,b)[p] +=y[i1]*(a1 - alph1)*np.inner(x[i1], x[p]) + y[i2]*(a2 - alph2)*np.inner(x[i2],x[p]) + b - b_new
    b = b_new
    return 1   
        
 

def ExExa(alpha,x,y,C,eps,b,i2,m,tol):
    y2 = y[i2]
    a2 = alpha[i2]
    E2 = Error_function(alpha,y,x,b)[i2] 
    r2 = E2*y2
    if ((r2 < -tol and a2 < C) or (r2 > tol and a2 > 0)):
        if (len(alpha[(alpha != 0) & (alpha != C)])) > 1:
            if Error_function(alpha,y,x,b)[i2] > 0:
                i1 = np.argmin(Error_function(alpha,y,x,b))
            elif Error_function(alpha,y,x,b)[i2]  <= 0:
                i1 = np.argmax(Error_function(alpha,y,x,b))
            if (TS(i1, i2, alpha,x,y,C,eps,b,m)):
                return 1
        t1 = list(range(m))
        random.shuffle(t1)
        for i1 in t1:
            if((alpha[i1] != 0) & (alpha[i1] != C)): 
                if TS(i1, i2, alpha,x,y,C,eps,b,m):
                    return 1
        t = list(range(m))
        random.shuffle(t)
        for i1 in t:
            if TS(i1, i2, alpha,x,y,C,eps,b,m):
                return 1 
    return 0

 
def Full_SMO(x,y,C,eps,m,tol):
    b=0.0
    alpha = np.zeros(x.shape[0])
    NC = 0
    EA = 1
    while (NC > 0 or EA):
        NC = 0
        if (EA):
            for i in range(alpha.shape[0]):
                NC += ExExa(alpha,x,y,C,eps,b,i,m,tol)
        else:
             for i in np.where((alpha != 0) & (alpha != C))[0]:
                NC += ExExa(alpha,x,y,C,eps,b,i,m,tol)
                
        if (EA == 1):
            EA = 0
        elif(NC == 0):
            EA = 1
    return alpha,b


def weights(alpha,X,y):
    m,n = X.shape
    w = np.zeros(n)
    for i in range(X.shape[0]):
        w += alpha[i]*y[i]*X[i,:]
    return w    

def get_scores(X,y,w,b):
    p = np.dot(X,w.T)+b
    m = y.shape[0]
    score = 0 
    for j in range(m):
        if (p[j] >= 0):
            p[j] = 1
        else :
            p[j] = -1
    for i in range(m):
        if (p[i]*y[i]) > 0 :
            score=score+1
    return score/m
    
    
start = timer()   
alpha,b = Full_SMO(text_x,text_t,1.29,0.0005,text_x.shape[0],0.0005)
print('Training score',get_scores(text_x,text_t,weights(alpha,text_x,text_t),b))


alpha1 , b1 = (Full_SMO(tp_x_1,tp_t_1,1.29,0.0005,tp_x_1.shape[0],0.0005))
w1 = weights(alpha1,tp_x_1,tp_t_1)
c = get_scores(tp_x_2,tp_t_2,w1,b1)
c+= get_scores(tp_x_3,tp_t_3,w1,b1)
alpha2 , b2 = (Full_SMO(tp_x_2,tp_t_2,1.29,0.0005,tp_x_2.shape[0],0.0005))
w2 = weights(alpha2,tp_x_2,tp_t_2)
c+= get_scores(tp_x_1,tp_t_1,w2,b2)
c+= get_scores(tp_x_3,tp_t_3,w2,b2)
alpha3 , b3 = (Full_SMO(tp_x_3,tp_t_3,1.29,0.0005,tp_x_3.shape[0],0.0005))
w3 = weights(alpha3,tp_x_3,tp_t_3)
c+= get_scores(tp_x_2,tp_t_2,w3,b3)
c+= get_scores(tp_x_1,tp_t_1,w3,b3)
print('Cross validation score',c/6)

end = timer()
print(end - start)




# In[ ]:


# FOR CLASSES {2,3}

#training data
text_x_2 = (dp[632:1230,:25])
text_t_2 = (dp[632:1230,25]).astype('int')



for i in range(text_t_2.shape[0]):
    if (text_t_2[i] == 2) :
        text_t_2[i] = 1
    else :
        text_t_2[i] = -1
 

 
#testing data
tp_x_1 = np.append(dp[632:732,:25],dp[943:1043,:25],axis=0)
tp_t_1 = np.append(dp[632:732,25],dp[943:1043,25],axis=0)
tp_t_1 = tp_t_1.astype('int')

for i in range(tp_t_1.shape[0]):
    if (tp_t_1[i] == 2) :
        tp_t_1[i] = 1
    else :
        tp_t_1[i] = -1
        
tp_x_2 = np.append(dp[732:832,:25],dp[1043:1143,:25],axis=0)
tp_t_2 = np.append(dp[732:832,25],dp[1043:1143,25],axis=0)
tp_t_2 = tp_t_2.astype('int')

for i in range(tp_t_2.shape[0]):
    if (tp_t_2[i] == 2) :
        tp_t_2[i] = 1
    else :
        tp_t_2[i] = -1
        
tp_x_3 = np.append(dp[832:942,:25],dp[1143:1230,:25],axis=0)
tp_t_3 = np.append(dp[832:942,25],dp[1143:1230,25],axis=0)
tp_t_3 = tp_t_3.astype('int')

for i in range(tp_t_3.shape[0]):
    if (tp_t_3[i] == 2) :
        tp_t_3[i] = 1
    else :
        tp_t_3[i] = -1
        
        
start = timer()   
alpha,b = Full_SMO(text_x_2,text_t_2,1.29,0.0005,text_x_2.shape[0],0.0005)
print('Training score',get_scores(text_x_2,text_t_2,weights(alpha,text_x_2,text_t_2),b))


alpha1 , b1 = (Full_SMO(tp_x_1,tp_t_1,1.29,0.0005,tp_x_1.shape[0],0.0005))
w1 = weights(alpha1,tp_x_1,tp_t_1)
c = get_scores(tp_x_2,tp_t_2,w1,b1)
c+= get_scores(tp_x_3,tp_t_3,w1,b1)
alpha2 , b2 = (Full_SMO(tp_x_2,tp_t_2,1.29,0.0005,tp_x_2.shape[0],0.0005))
w2 = weights(alpha2,tp_x_2,tp_t_2)
c+= get_scores(tp_x_1,tp_t_1,w2,b2)
c+= get_scores(tp_x_3,tp_t_3,w2,b2)
alpha3 , b3 = (Full_SMO(tp_x_3,tp_t_3,1.29,0.0005,tp_x_3.shape[0],0.0005))
w3 = weights(alpha3,tp_x_3,tp_t_3)
c+= get_scores(tp_x_2,tp_t_2,w3,b3)
c+= get_scores(tp_x_1,tp_t_1,w3,b3)
print('Cross validation score',c/6)

end = timer()
print(end - start)


# In[ ]:


# FOR CLASSES {4,5}

#training data
text_x_3 = dp[1232:1800,:25]
text_t_3 = dp[1232:1800,25].astype('int')

for i in range(text_t_3.shape[0]):
    if (text_t_3[i] == 4) :
        text_t_3[i] = 1
    else :
        text_t_3[i] = -1

#testing data
tp_x_1 = np.append(dp[1232:1332,:25],dp[1533:1610,:25],axis=0)
tp_t_1 = np.append(dp[1232:1332,25],dp[1533:1610,25],axis=0)
tp_t_1 = tp_t_1.astype('int')

for i in range(tp_t_1.shape[0]):
    if (tp_t_1[i] == 4) :
        tp_t_1[i] = 1
    else :
        tp_t_1[i] = -1
        
tp_x_2 = np.append(dp[1333:1433,:25],dp[1610:1699,:25],axis=0)
tp_t_2 = np.append(dp[1333:1433,25],dp[1610:1699,25],axis=0)
tp_t_2 = tp_t_2.astype('int')

for i in range(tp_t_2.shape[0]):
    if (tp_t_2[i] == 4) :
        tp_t_2[i] = 1
    else :
        tp_t_2[i] = -1
        
tp_x_3 = np.append(dp[1433:1532,:25],dp[1700:1800,:25],axis=0)
tp_t_3 = np.append(dp[1433:1532,25],dp[1700:1800,25],axis=0)
tp_t_3 = tp_t_3.astype('int')

for i in range(tp_t_3.shape[0]):
    if (tp_t_3[i] == 4) :
        tp_t_3[i] = 1
    else :
        tp_t_3[i] = -1
        
start = timer()   
alpha,b = Full_SMO(text_x_3,text_t_3,1.29,0.0005,text_x_3.shape[0],0.0005)
print('Training score',get_scores(text_x_3,text_t_3,weights(alpha,text_x_3,text_t_3),b))


alpha1 , b1 = (Full_SMO(tp_x_1,tp_t_1,1.29,0.0005,tp_x_1.shape[0],0.0005))
w1 = weights(alpha1,tp_x_1,tp_t_1)
c = get_scores(tp_x_2,tp_t_2,w1,b1)
c+= get_scores(tp_x_3,tp_t_3,w1,b1)
alpha2 , b2 = (Full_SMO(tp_x_2,tp_t_2,1.29,0.0005,tp_x_2.shape[0],0.0005))
w2 = weights(alpha2,tp_x_2,tp_t_2)
c+= get_scores(tp_x_1,tp_t_1,w2,b2)
c+= get_scores(tp_x_3,tp_t_3,w2,b2)
alpha3 , b3 = (Full_SMO(tp_x_3,tp_t_3,1.29,0.0005,tp_x_3.shape[0],0.0005))
w3 = weights(alpha3,tp_x_3,tp_t_3)
c+= get_scores(tp_x_2,tp_t_2,w3,b3)
c+= get_scores(tp_x_1,tp_t_1,w3,b3)
print('Cross validation score',c/6)

end = timer()
print(end - start)

