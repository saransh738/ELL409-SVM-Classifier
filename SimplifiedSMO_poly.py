#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
        
start = timer()

def computing_F(i,b,alpha,X,Y,gamma,degree):
    sum=0
    for j in range(X.shape[0]):
        sum += (alpha[j]*Y[j])*((1+gamma*(np.inner(X[j],X[i])))**degree)
                                
    return sum+b


def randomy(m,i):
    j = random.randint(0, m)
    while (i == j):
        j = random.randint(0, m)
    return j
    
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

def computing_neta(Xi,Xj,gamma,degree):
    return 2*((1+gamma*(np.inner(Xj,Xi)))**degree)-((1+gamma*(np.inner(Xi,Xi)))**degree)-((1+gamma*(np.inner(Xj,Xj)))**degree)

def computing_alpha_j(neta,yj,Ei,Ej,alpha_j):
    return alpha_j-((yj)*(Ei-Ej))/neta
     
def cliping_alpha_j(L,H,alpha_j):
    if(alpha_j>H):
        return H
    elif(alpha_j<L):
        return L
    return alpha_j

        
def computing_b2(b,Xi,Xj,yi,yj,alpha_j,alpha_i,alphaold_i,alphaold_j,Ei,Ej,gamma,degree): 
    return (b - Ej - yi) * (alpha_i - alphaold_i) *((1+gamma*(np.inner(Xi,Xj)))**degree) - (yj * (alpha_j - alphaold_j)) *((1+gamma*(np.inner(Xj,Xj)))**degree) 

def computing_b1(b,Xi,Xj,yi,yj,alpha_j,alpha_i,alphaold_i,alphaold_j,Ei,Ej,gamma,degree): 
    return (b - Ei - yi) * (alpha_i - alphaold_i) * ((1+gamma*(np.inner(Xi,Xi)))**degree) - (yj * (alpha_j - alphaold_j)) * ((1+gamma*(np.inner(Xi,Xj)))**degree) 

def computing_b(C,alpha_j,alpha_i,b1,b2):    
    if(alpha_i<C and alpha_i>0):
        return b1
    elif(alpha_j<C and alpha_j>0):
        return b2
    else:
        return (b1+b2)/2
   
    
def simplified_SMO(C,gamma,degree,tol,max_iter,X,y):
    alpha = np.zeros(X.shape[0])
    b = 0
    iter = 0 
    while(iter<max_iter):
        num_changed_alphas = 0
        for i in range(X.shape[0]):
            Ei = computing_F(i,b,alpha,X,y,gamma,degree) - y[i]
            if(((y[i]*Ei)<(-1*tol) and alpha[i]<C) or ((y[i]*Ei)>tol and alpha[i]>0)):
                j = randomy(X.shape[0]-1,i)
                Ej = computing_F(j,b,alpha,X,y,gamma,degree) - y[j]
                alphaolds_j = alpha[j]
                alphaolds_i = alpha[i]
                L =  computing_L(alpha[i],alpha[j],C,y[i],y[j])
                H =  computing_H(alpha[i],alpha[j],C,y[i],y[j])
                if(L==H):
                    continue
                neta = computing_neta(X[i],X[j],gamma,degree)    
                if(neta>=0):
                    continue 
                alpha[j] = cliping_alpha_j(L,H, computing_alpha_j(neta,y[j],Ei,Ej,alpha[j]))
                if(np.abs(alpha[j]-alphaolds_j)<tol):   
                    continue
                alpha[i] = alpha[i] + (y[i] * y[j]) * (alphaolds_j - alpha[j])
                b1 = computing_b1(b,X[i],X[j],y[i],y[j],alpha[j],alpha[i],alphaolds_i,alphaolds_j,Ei,Ej,gamma,degree)
                b2 = computing_b2(b,X[i],X[j],y[i],y[j],alpha[j],alpha[i],alphaolds_i,alphaolds_j,Ei,Ej,gamma,degree)
                b = computing_b(C,alpha[j],alpha[i],b1,b2)
                num_changed_alphas += 1
        if (num_changed_alphas == 0):
            iter += 1
        else:
            iter = 0
    return alpha ,b
      
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

alpha , b = (simplified_SMO(1,1,1,0.0005,1,text_x,text_t))  
print('Training score',get_scores(text_x,text_t,weights(alpha,text_x,text_t),b))


alpha1 , b1 = (simplified_SMO(1,1,1,0.0005,1,tp_x_1,tp_t_1))
w1 = weights(alpha1,tp_x_1,tp_t_1)
c = get_scores(tp_x_2,tp_t_2,w1,b1)
c+= get_scores(tp_x_3,tp_t_3,w1,b1)
alpha2 , b2 = (simplified_SMO(1,1,1,0.0005,1,tp_x_2,tp_t_2))
w2 = weights(alpha2,tp_x_2,tp_t_2)
c+= get_scores(tp_x_1,tp_t_1,w2,b2)
c+= get_scores(tp_x_3,tp_t_3,w2,b2)
alpha3 , b3 = (simplified_SMO(1,1,1,0.0005,1,tp_x_3,tp_t_3))
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
    
start1 = timer()    
alpha , b = (simplified_SMO(1,1,5,0.0005,500,text_x_2,text_t_2))  
print('Training score',get_scores(text_x_2,text_t_2,weights(alpha,text_x_2,text_t_2),b))


alpha1 , b1 = (simplified_SMO(1,1,5,0.0005,500,tp_x_1,tp_t_1))
w1 = weights(alpha1,tp_x_1,tp_t_1)
c1 = get_scores(tp_x_2,tp_t_2,w1,b1)
c1+= get_scores(tp_x_3,tp_t_3,w1,b1)
alpha2 , b2 = (simplified_SMO(1,1,5,0.0005,500,tp_x_2,tp_t_2))
w2 = weights(alpha2,tp_x_2,tp_t_2)
c1+= get_scores(tp_x_1,tp_t_1,w2,b2)
c1+= get_scores(tp_x_3,tp_t_3,w2,b2)
alpha3 , b3 = (simplified_SMO(1,1,5,0.0005,500,tp_x_3,tp_t_3))
w3 = weights(alpha3,tp_x_3,tp_t_3)
c1+= get_scores(tp_x_2,tp_t_2,w3,b3)
c1+= get_scores(tp_x_1,tp_t_1,w3,b3)
print('Cross validation score',c1/6)

end1 = timer()
print(end1 - start1)        


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

start2 = timer()
alpha , b = (simplified_SMO(1,1.29,1,0.0005,2,text_x_3,text_t_3))  
print('Training score',get_scores(text_x_3,text_t_3,weights(alpha,text_x_3,text_t_3),b))


alpha1 , b1 = (simplified_SMO(1,1.29,1,0.0005,2,tp_x_1,tp_t_1))
w1 = weights(alpha1,tp_x_1,tp_t_1)
c2 = get_scores(tp_x_2,tp_t_2,w1,b1)
c2+= get_scores(tp_x_3,tp_t_3,w1,b1)
alpha2 , b2 = (simplified_SMO(1,1.29,1,0.0005,2,tp_x_2,tp_t_2))
w2 = weights(alpha2,tp_x_2,tp_t_2)
c2+= get_scores(tp_x_1,tp_t_1,w2,b2)
c2+= get_scores(tp_x_3,tp_t_3,w2,b2)
alpha3 , b3 = (simplified_SMO(1,1.29,1,0.0005,2,tp_x_3,tp_t_3))
w3 = weights(alpha3,tp_x_3,tp_t_3)
c2+= get_scores(tp_x_2,tp_t_2,w3,b3)
c2+= get_scores(tp_x_1,tp_t_1,w3,b3)
print('Cross validation score',c2/6)

end2 = timer()
print(end2 - start2)  


# In[ ]:




