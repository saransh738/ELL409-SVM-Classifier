#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libsvm.svmutil import *
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
import cvxopt
from timeit import default_timer as timer

#Reading files
data_points = pd.read_csv('2019MT60763.csv', header = None, nrows = 3000)
data = np.array((data_points.sort_values(data_points.columns[25])).values)
dp = np.array(data)
class_label = dp[:,25]

# counting no of occurence of labels of each class
unique, counts = np.unique(class_label, return_counts=True)
dict(zip(unique, counts))
#print(counts)

# for 25 features 

# FOR CLASSES {0,1}

text_x = dp[:631,:25]
text_t = dp[:631,25].astype('int')

for i in range(text_t.shape[0]):
    if (text_t[i] == 0) :
        text_t[i] = 1
    else :
        text_t[i] = -1
        
#testing data
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

#function to compute kernel function 
def compute_K(kernel,X,gamma,degree):
    K = X.dot(np.transpose(X))
    if(kernel == 'poly'):
        K = (gamma*K+1)**degree
    elif(kernel == 'rbf'):
        u = np.diag(X.dot(np.transpose(X))).reshape((-1, 1))*np.ones((1, X.shape[0]))
        K = 2*K-u- np.diag(X.dot(np.transpose(X))).reshape((1, -1))*np.ones((X.shape[0], 1))
        K = np.exp(gamma*K)
    elif(kernel == 'sigmoid'):
        K = np.tanh(gamma*K+1)
    return K

def cvx_fiting(C,X,y,K):
    n = X.shape[0]
    y = y.reshape((-1,1)) * 1.0
    H = ((y.dot(np.transpose(y)))*K)
    Q = cvxopt.matrix(-np.ones((n,1)))
    p = cvxopt.matrix(H)
    G = cvxopt.matrix(np.concatenate((-np.eye(n), np.eye(n))))
    h = cvxopt.matrix(np.append(np.zeros((n,1)),(np.ones((n,1)))*C))
    A = cvxopt.matrix(np.transpose(y))
    b = cvxopt.matrix(0.0)
    cvxopt.solvers.options['show_progress'] = False
    sol=cvxopt.solvers.qp(p,Q,G,h,A,b)
    multipliers = np.array(sol['x'])
    return multipliers

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

def weights(alpha,X,y):
    m,n = X.shape
    w = np.zeros(n)
    for i in range(X.shape[0]):
        w += alpha[i]*y[i]*X[i,:]
    return w  

support_vectors = np.where(cvx_fiting(1.0,text_x,(text_t),compute_K('linear',text_x,0,0)) > 1e-4)[0]
print(support_vectors)                                
support_vectors = np.where(cvx_fiting(1.29,text_x,(text_t),compute_K('rbf',text_x,1.0,0)) > 1e-4)[0]
print(support_vectors)                                
support_vectors = np.where(cvx_fiting(1.0,text_x,(text_t),compute_K('poly',text_x,1.0,1)) > 1e-4)[0]
print(support_vectors)                                
                       
  

start = timer()
w = weights((cvx_fiting(1.0,text_x,text_t,compute_K('linear',text_x,0,0))),text_x,text_t)
b = text_t[((cvx_fiting(1.0,text_x,text_t,compute_K('linear',text_x,0,0))) > 1e-4).reshape(-1)] - np.dot(text_x[((cvx_fiting(1.0,text_x,text_t,compute_K('linear',text_x,0,0))) > 1e-4).reshape(-1)], w)
print('Training score',get_scores(text_x,text_t,w,b[0])) 
w1 = weights((cvx_fiting(1.0,tp_x_1,tp_t_1,compute_K('linear',tp_x_1,0,0))),tp_x_1,tp_t_1)
b1 = tp_t_1[((cvx_fiting(1.0,tp_x_1,tp_t_1,compute_K('linear',tp_x_1,0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_1[((cvx_fiting(1.0,tp_x_1,tp_t_1,compute_K('linear',tp_x_1,0,0))) > 1e-4).reshape(-1)], w1)
p1 = get_scores(tp_x_2,tp_t_2,w1,b1[0])
p1+=get_scores(tp_x_3,tp_t_3,w1,b1[0])
w2 = weights((cvx_fiting(1.0,tp_x_3,tp_t_3,compute_K('linear',tp_x_3,0,0))),tp_x_3,tp_t_3)
b2 = tp_t_3[((cvx_fiting(1.0,tp_x_3,tp_t_3,compute_K('linear',tp_x_3,0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_3[((cvx_fiting(1.0,tp_x_3,tp_t_3,compute_K('linear',tp_x_3,0,0))) > 1e-4).reshape(-1)], w2)
p1+=get_scores(tp_x_2,tp_t_2,w2,b2[0])
p1+= get_scores(tp_x_1,tp_t_1,w2,b2[0])
w3 = weights((cvx_fiting(1.0,tp_x_2,tp_t_2,compute_K('linear',tp_x_2,0,0))),tp_x_2,tp_t_2)
b3 = tp_t_2[((cvx_fiting(1.0,tp_x_2,tp_t_2,compute_K('linear',tp_x_2,0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_2[((cvx_fiting(1.0,tp_x_2,tp_t_2,compute_K('linear',tp_x_2,0,0))) > 1e-4).reshape(-1)], w3)
p1+= get_scores(tp_x_1,tp_t_1,w,b[0])
p1+= get_scores(tp_x_3,tp_t_3,w,b[0])
print('Cross_validation score',p1/6) 
end = timer()
print('Time',end - start)

w = weights((cvx_fiting(1.29,text_x,text_t,compute_K('rbf',text_x,1.0,0))),text_x,text_t)
b = text_t[((cvx_fiting(1.29,text_x,text_t,compute_K('rbf',text_x,1.0,0))) > 1e-4).reshape(-1)] - np.dot(text_x[((cvx_fiting(1.29,text_x,text_t,compute_K('rbf',text_x,1.0,0))) > 1e-4).reshape(-1)], w)
print('Training score',get_scores(text_x,text_t,w,b[0])) 
w1 = weights((cvx_fiting(1.29,tp_x_1,tp_t_1,compute_K('rbf',tp_x_1,1.0,0))),tp_x_1,tp_t_1)
b1 = tp_t_1[((cvx_fiting(1.29,tp_x_1,tp_t_1,compute_K('rbf',tp_x_1,1.0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_1[((cvx_fiting(1.29,tp_x_1,tp_t_1,compute_K('rbf',tp_x_1,1.0,0))) > 1e-4).reshape(-1)], w1)
p8 = get_scores(tp_x_2,tp_t_2,w1,b1[0])
p8+=get_scores(tp_x_3,tp_t_3,w1,b1[0])
w2 = weights((cvx_fiting(1.29,tp_x_3,tp_t_3,compute_K('rbf',tp_x_3,1.0,0))),tp_x_3,tp_t_3)
b2 = tp_t_3[((cvx_fiting(1.29,tp_x_3,tp_t_3,compute_K('rbf',tp_x_3,1.0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_3[((cvx_fiting(1.29,tp_x_3,tp_t_3,compute_K('rbf',tp_x_3,1.0,0))) > 1e-4).reshape(-1)], w2)
p8+=get_scores(tp_x_2,tp_t_2,w2,b2[0])
p8+= get_scores(tp_x_1,tp_t_1,w2,b2[0])
w3 = weights((cvx_fiting(1.29,tp_x_2,tp_t_2,compute_K('rbf',tp_x_2,1.0,0))),tp_x_2,tp_t_2)
b3 = tp_t_2[((cvx_fiting(1.29,tp_x_2,tp_t_2,compute_K('rbf',tp_x_2,1.0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_2[((cvx_fiting(1.29,tp_x_2,tp_t_2,compute_K('rbf',tp_x_2,1.0,0))) > 1e-4).reshape(-1)], w3)
p8+= get_scores(tp_x_1,tp_t_1,w3,b3[0])
p8+= get_scores(tp_x_3,tp_t_3,w3,b3[0])
print('Cross_validation score',p8/6)

start1 = timer()
w = weights((cvx_fiting(1.0,text_x,text_t,compute_K('poly',text_x,1.0,1))),text_x,text_t)
b= text_t[((cvx_fiting(1.0,text_x,text_t,compute_K('poly',text_x,1.0,1))) > 1e-4).reshape(-1)] - np.dot(text_x[((cvx_fiting(1.0,text_x,text_t,compute_K('poly',text_x,1.0,1))) > 1e-4).reshape(-1)], w)
print('Training score',get_scores(text_x,text_t,w,b[0])) 
w1 = weights((cvx_fiting(1.0,tp_x_1,tp_t_1,compute_K('poly',tp_x_1,1.0,1))),tp_x_1,tp_t_1)
b1 = tp_t_1[((cvx_fiting(1.0,tp_x_1,tp_t_1,compute_K('poly',tp_x_1,1.0,1))) > 1e-4).reshape(-1)] - np.dot(tp_x_1[((cvx_fiting(1.0,tp_x_1,tp_t_1,compute_K('poly',tp_x_1,1.0,1))) > 1e-4).reshape(-1)], w1)
p4 = get_scores(tp_x_2,tp_t_2,w1,b1[0])
p4+=get_scores(tp_x_3,tp_t_3,w1,b1[0])
w2 = weights((cvx_fiting(1.0,tp_x_3,tp_t_3,compute_K('poly',tp_x_3,1.0,1))),tp_x_3,tp_t_3)
b2 = tp_t_3[((cvx_fiting(1.0,tp_x_3,tp_t_3,compute_K('poly',tp_x_3,1.0,1))) > 1e-4).reshape(-1)] - np.dot(tp_x_3[((cvx_fiting(1.0,tp_x_3,tp_t_3,compute_K('poly',tp_x_3,1.0,1))) > 1e-4).reshape(-1)], w2)
p4+=get_scores(tp_x_2,tp_t_2,w2,b2[0])
p4+= get_scores(tp_x_1,tp_t_1,w2,b2[0])
w3 = weights((cvx_fiting(1.0,tp_x_2,tp_t_2,compute_K('poly',tp_x_2,1.0,1))),tp_x_2,tp_t_2)
b3 = tp_t_2[((cvx_fiting(1.0,tp_x_2,tp_t_2,compute_K('poly',tp_x_2,1.0,1))) > 1e-4).reshape(-1)] - np.dot(tp_x_2[((cvx_fiting(1.0,tp_x_2,tp_t_2,compute_K('pol',tp_x_2,1.0,1))) > 1e-4).reshape(-1)], w3)
p4+= get_scores(tp_x_1,tp_t_1,w3,b3[0])
p4+= get_scores(tp_x_3,tp_t_3,w3,b3[0])
print('Cross_validation score',p4/6)     
end1 = timer()
print('TIME',end1 - start1)


# In[2]:


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
        
support_vectors = np.where(cvx_fiting(7.74,text_x_2,(text_t_2),compute_K('linear',text_x_2,0,0)) > 1e-4)[0]
print(support_vectors)                                
support_vectors = np.where(cvx_fiting(1.29,text_x_2,(text_t_2),compute_K('rbf',text_x_2,1.0,0)) > 1e-4)[0]
print(support_vectors)                                
support_vectors = np.where(cvx_fiting(1.0,text_x_2,(text_t_2),compute_K('poly',text_x_2,1.0,5)) > 1e-9)[0]
print(support_vectors)   

start3 = timer()
w = weights((cvx_fiting(7.74,text_x_2,text_t_2,compute_K('linear',text_x_2,0,0))),text_x_2,text_t_2)
b = text_t_2[((cvx_fiting(7.74,text_x_2,text_t_2,compute_K('linear',text_x_2,0,0))) > 1e-4).reshape(-1)] - np.dot(text_x_2[((cvx_fiting(7.74,text_x_2,text_t_2,compute_K('linear',text_x_2,0,0))) > 1e-4).reshape(-1)], w)
print('Training score',get_scores(text_x_2,text_t_2,w,b[0])) 
w1 = weights((cvx_fiting(7.74,tp_x_1,tp_t_1,compute_K('linear',tp_x_1,0,0))),tp_x_1,tp_t_1)
b1 = tp_t_1[((cvx_fiting(7.74,tp_x_1,tp_t_1,compute_K('linear',tp_x_1,0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_1[((cvx_fiting(7.74,tp_x_1,tp_t_1,compute_K('linear',tp_x_1,0,0))) > 1e-4).reshape(-1)], w1)
p2 = get_scores(tp_x_2,tp_t_2,w1,b1[0])
p2+=get_scores(tp_x_3,tp_t_3,w1,b1[0])
w2 = weights((cvx_fiting(7.74,tp_x_3,tp_t_3,compute_K('linear',tp_x_3,0,0))),tp_x_3,tp_t_3)
b2 = tp_t_3[((cvx_fiting(7.74,tp_x_3,tp_t_3,compute_K('linear',tp_x_3,0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_3[((cvx_fiting(7.74,tp_x_3,tp_t_3,compute_K('linear',tp_x_3,0,0))) > 1e-4).reshape(-1)], w2)
p2+=get_scores(tp_x_2,tp_t_2,w2,b2[0])
p2+= get_scores(tp_x_1,tp_t_1,w2,b2[0])
w3 = weights((cvx_fiting(7.74,tp_x_2,tp_t_2,compute_K('linear',tp_x_2,0,0))),tp_x_2,tp_t_2)
b3 = tp_t_2[((cvx_fiting(7.74,tp_x_2,tp_t_2,compute_K('linear',tp_x_2,0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_2[((cvx_fiting(7.74,tp_x_2,tp_t_2,compute_K('linear',tp_x_2,0,0))) > 1e-4).reshape(-1)], w3)
p2+= get_scores(tp_x_1,tp_t_1,w,b[0])
p2+= get_scores(tp_x_3,tp_t_3,w,b[0])
print('Cross_validation score',p2/6)   
end3 = timer()
print('TIME',end3 - start3)

w = weights((cvx_fiting(1.29,text_x_2,text_t_2,compute_K('rbf',text_x_2,1.0,0))),text_x_2,text_t_2)
b = text_t_2[((cvx_fiting(1.29,text_x_2,text_t_2,compute_K('rbf',text_x_2,1.0,0))) > 1e-4).reshape(-1)] - np.dot(text_x_2[((cvx_fiting(1.29,text_x_2,text_t_2,compute_K('rbf',text_x_2,1.0,0))) > 1e-4).reshape(-1)], w)
print('Training score',get_scores(text_x_2,text_t_2,w,b[0])) 
w1 = weights((cvx_fiting(1.29,tp_x_1,tp_t_1,compute_K('rbf',tp_x_1,1.0,0))),tp_x_1,tp_t_1)
b1 = tp_t_1[((cvx_fiting(1.29,tp_x_1,tp_t_1,compute_K('rbf',tp_x_1,1.0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_1[((cvx_fiting(1.29,tp_x_1,tp_t_1,compute_K('rbf',tp_x_1,1.0,0))) > 1e-4).reshape(-1)], w1)
p7 = get_scores(tp_x_2,tp_t_2,w1,b1[0])
p7+=get_scores(tp_x_3,tp_t_3,w1,b1[0])
w2 = weights((cvx_fiting(1.29,tp_x_3,tp_t_3,compute_K('rbf',tp_x_3,1.0,0))),tp_x_3,tp_t_3)
b2 = tp_t_3[((cvx_fiting(1.29,tp_x_3,tp_t_3,compute_K('rbf',tp_x_3,1.0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_3[((cvx_fiting(1.29,tp_x_3,tp_t_3,compute_K('rbf',tp_x_3,1.0,0))) > 1e-4).reshape(-1)], w2)
p7+=get_scores(tp_x_2,tp_t_2,w2,b2[0])
p7+= get_scores(tp_x_1,tp_t_1,w2,b2[0])
w3 = weights((cvx_fiting(1.29,tp_x_2,tp_t_2,compute_K('rbf',tp_x_2,1.0,0))),tp_x_2,tp_t_2)
b3 = tp_t_2[((cvx_fiting(1.29,tp_x_2,tp_t_2,compute_K('rbf',tp_x_2,1.0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_2[((cvx_fiting(1.29,tp_x_2,tp_t_2,compute_K('rbf',tp_x_2,1.0,0))) > 1e-4).reshape(-1)], w3)
p7+= get_scores(tp_x_1,tp_t_1,w3,b3[0])
p7+= get_scores(tp_x_3,tp_t_3,w3,b3[0])
print('Cross_validation score',p7/6)
     
start4 = timer()    
w = weights((cvx_fiting(1.0,text_x_2,text_t_2,compute_K('poly',text_x_2,1.0,5))),text_x_2,text_t_2)
b = text_t_2[((cvx_fiting(1.0,text_x_2,text_t_2,compute_K('poly',text_x_2,1.0,5))) > 1e-9).reshape(-1)] - np.dot(text_x_2[((cvx_fiting(1.0,text_x_2,text_t_2,compute_K('poly',text_x_2,1.0,5))) > 1e-9).reshape(-1)], w)
print('Training score',get_scores(text_x_2,text_t_2,w,b[0])) 
w1 = weights((cvx_fiting(1.0,tp_x_1,tp_t_1,compute_K('poly',tp_x_1,1.0,5))),tp_x_1,tp_t_1)
b1 = tp_t_1[((cvx_fiting(1.0,tp_x_1,tp_t_1,compute_K('poly',tp_x_1,1.0,5))) > 1e-9).reshape(-1)] - np.dot(tp_x_1[((cvx_fiting(1.0,tp_x_1,tp_t_1,compute_K('poly',tp_x_1,1.0,5))) > 1e-9).reshape(-1)], w1)
p5 = get_scores(tp_x_2,tp_t_2,w1,b1[0])
p5+=get_scores(tp_x_3,tp_t_3,w1,b1[0])
w2 = weights((cvx_fiting(1.0,tp_x_3,tp_t_3,compute_K('poly',tp_x_3,1.0,5))),tp_x_3,tp_t_3)
b2 = tp_t_3[((cvx_fiting(1.0,tp_x_3,tp_t_3,compute_K('poly',tp_x_3,1.0,5))) > 1e-9).reshape(-1)] - np.dot(tp_x_3[((cvx_fiting(1.0,tp_x_3,tp_t_3,compute_K('poly',tp_x_3,1.0,5))) > 1e-9).reshape(-1)], w2)
p5+=get_scores(tp_x_2,tp_t_2,w2,b2[0])
p5+= get_scores(tp_x_1,tp_t_1,w2,b2[0])
w3 = weights((cvx_fiting(1.0,tp_x_2,tp_t_2,compute_K('poly',tp_x_2,1.0,5))),tp_x_2,tp_t_2)
b3 = tp_t_2[((cvx_fiting(1.0,tp_x_2,tp_t_2,compute_K('poly',tp_x_2,1.0,5))) > 1e-9).reshape(-1)] - np.dot(tp_x_2[((cvx_fiting(1.0,tp_x_2,tp_t_2,compute_K('poly',tp_x_2,1.0,5))) > 1e-9).reshape(-1)], w3)
p5+= get_scores(tp_x_1,tp_t_1,w3,b3[0])
p5+= get_scores(tp_x_3,tp_t_3,w3,b3[0])
print('Cross_validation score',p5/6)         
end4 = timer()
print('TIME',end4 - start4)


# In[35]:


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
        
support_vectors = np.where(cvx_fiting(1.29,text_x_3,(text_t_3),compute_K('linear',text_x_3,0,0)) > 1e-4)[0]
print(support_vectors)                                
support_vectors = np.where(cvx_fiting(1.29,text_x_3,(text_t_3),compute_K('rbf',text_x_3,1.0,0)) > 1e-4)[0]
print(support_vectors)                                
support_vectors = np.where(cvx_fiting(1.0,text_x_3,(text_t_3),compute_K('poly',text_x_3,1.29,1)) > 1e-4)[0]
print(support_vectors)           

start7 = timer()
w = weights((cvx_fiting(1.29,text_x_3,text_t_3,compute_K('linear',text_x_3,0,0))),text_x_3,text_t_3)
b = text_t_3[((cvx_fiting(1.29,text_x_3,text_t_3,compute_K('linear',text_x_3,0,0))) > 1e-4).reshape(-1)] - np.dot(text_x_3[((cvx_fiting(1.29,text_x_3,text_t_3,compute_K('linear',text_x_3,0,0))) > 1e-4).reshape(-1)], w)
print('Training score',get_scores(text_x_3,text_t_3,w,b[0])) 
w1 = weights((cvx_fiting(1.29,tp_x_1,tp_t_1,compute_K('linear',tp_x_1,0,0))),tp_x_1,tp_t_1)
b1 = tp_t_1[((cvx_fiting(1.29,tp_x_1,tp_t_1,compute_K('linear',tp_x_1,0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_1[((cvx_fiting(1.29,tp_x_1,tp_t_1,compute_K('linear',tp_x_1,0,0))) > 1e-4).reshape(-1)], w1)
p5 = get_scores(tp_x_2,tp_t_2,w1,b1[0])
p5+=get_scores(tp_x_3,tp_t_3,w1,b1[0])
w2 = weights((cvx_fiting(1.29,tp_x_3,tp_t_3,compute_K('linear',tp_x_3,0,0))),tp_x_3,tp_t_3)
b2 = tp_t_3[((cvx_fiting(1.29,tp_x_3,tp_t_3,compute_K('linear',tp_x_3,0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_3[((cvx_fiting(1.29,tp_x_3,tp_t_3,compute_K('linear',tp_x_3,0,0))) > 1e-4).reshape(-1)], w2)
p5+=get_scores(tp_x_2,tp_t_2,w2,b2[0])
p5+= get_scores(tp_x_1,tp_t_1,w2,b2[0])
w3 = weights((cvx_fiting(1.29,tp_x_2,tp_t_2,compute_K('linear',tp_x_2,0,0))),tp_x_2,tp_t_2)
b3 = tp_t_2[((cvx_fiting(1.29,tp_x_2,tp_t_2,compute_K('linear',tp_x_2,0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_2[((cvx_fiting(1.29,tp_x_2,tp_t_2,compute_K('linear',tp_x_2,0,0))) > 1e-4).reshape(-1)], w3)
p5+= get_scores(tp_x_1,tp_t_1,w,b[0])
p5+= get_scores(tp_x_3,tp_t_3,w,b[0])
print('Cross_validation score',p5/6)   
end7 = timer()
print('TIME',end7 - start7)

w4 = weights((cvx_fiting(1.29,text_x_3,text_t_3,compute_K('rbf',text_x_3,1.0,0))),text_x_3,text_t_3)
b4 = text_t_3[((cvx_fiting(1.29,text_x_3,text_t_3,compute_K('rbf',text_x_3,1.0,0))) > 1e-4).reshape(-1)] - np.dot(text_x_3[((cvx_fiting(1.29,text_x_3,text_t_3,compute_K('rbf',text_x_3,1.0,0))) > 1e-4).reshape(-1)], w4)
print('Training score',get_scores(text_x_3,text_t_3,w4,b4[0])) 
w5 = weights((cvx_fiting(1.29,tp_x_1,tp_t_1,compute_K('rbf',tp_x_1,1.0,0))),tp_x_1,tp_t_1)
b5 = tp_t_1[((cvx_fiting(1.29,tp_x_1,tp_t_1,compute_K('rbf',tp_x_1,1.0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_1[((cvx_fiting(1.29,tp_x_1,tp_t_1,compute_K('rbf',tp_x_1,1.0,0))) > 1e-4).reshape(-1)], w5)
p5 = get_scores(tp_x_2,tp_t_2,w5,b5[0])
p5+=get_scores(tp_x_3,tp_t_3,w5,b5[0])
w6 = weights((cvx_fiting(1.29,tp_x_3,tp_t_3,compute_K('rbf',tp_x_3,1.0,0))),tp_x_3,tp_t_3)
b6 = tp_t_3[((cvx_fiting(1.29,tp_x_3,tp_t_3,compute_K('rbf',tp_x_3,1.0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_3[((cvx_fiting(1.29,tp_x_3,tp_t_3,compute_K('rbf',tp_x_3,1.0,0))) > 1e-4).reshape(-1)], w6)
p5+=get_scores(tp_x_2,tp_t_2,w6,b6[0])
p5+= get_scores(tp_x_1,tp_t_1,w6,b6[0])
w7 = weights((cvx_fiting(1.29,tp_x_2,tp_t_2,compute_K('rbf',tp_x_2,1.0,0))),tp_x_2,tp_t_2)
b7 = tp_t_2[((cvx_fiting(1.29,tp_x_2,tp_t_2,compute_K('rbf',tp_x_2,1.0,0))) > 1e-4).reshape(-1)] - np.dot(tp_x_2[((cvx_fiting(1.29,tp_x_2,tp_t_2,compute_K('rbf',tp_x_2,1.0,0))) > 1e-4).reshape(-1)], w7)
p5+= get_scores(tp_x_1,tp_t_1,w7,b7[0])
p5+= get_scores(tp_x_3,tp_t_3,w7,b7[0])
print('Cross_validation score',p5/6)

start6 = timer()
w = weights((cvx_fiting(1.0,text_x_3,text_t_3,compute_K('poly',text_x_3,1.29,1))),text_x_3,text_t_3)
b = text_t_3[((cvx_fiting(1.0,text_x_3,text_t_3,compute_K('poly',text_x_3,1.29,1))) > 1e-9).reshape(-1)] - np.dot(text_x_3[((cvx_fiting(1.0,text_x_3,text_t_3,compute_K('poly',text_x_3,1.29,1))) > 1e-9).reshape(-1)], w)
print('Training score',get_scores(text_x_3,text_t_3,w,b[0])) 
w1 = weights((cvx_fiting(1.0,tp_x_1,tp_t_1,compute_K('poly',tp_x_1,1.29,1))),tp_x_1,tp_t_1)
b1 = tp_t_1[((cvx_fiting(1.0,tp_x_1,tp_t_1,compute_K('poly',tp_x_1,1.29,1))) > 1e-9).reshape(-1)] - np.dot(tp_x_1[((cvx_fiting(1.0,tp_x_1,tp_t_1,compute_K('poly',tp_x_1,1.29,1))) > 1e-9).reshape(-1)], w1)
p6 = get_scores(tp_x_2,tp_t_2,w1,b1[0])
p6+=get_scores(tp_x_3,tp_t_3,w1,b1[0])
w2 = weights((cvx_fiting(1.0,tp_x_3,tp_t_3,compute_K('poly',tp_x_3,1.29,1))),tp_x_3,tp_t_3)
b2 = tp_t_3[((cvx_fiting(1.0,tp_x_3,tp_t_3,compute_K('poly',tp_x_3,1.29,1))) > 1e-9).reshape(-1)] - np.dot(tp_x_3[((cvx_fiting(1.0,tp_x_3,tp_t_3,compute_K('poly',tp_x_3,1.29,1))) > 1e-9).reshape(-1)], w2)
p6+=get_scores(tp_x_2,tp_t_2,w2,b2[0])
p6+= get_scores(tp_x_1,tp_t_1,w2,b2[0])
w3 = weights((cvx_fiting(1.0,tp_x_2,tp_t_2,compute_K('poly',tp_x_2,1.29,1))),tp_x_2,tp_t_2)
b3 = tp_t_2[((cvx_fiting(1.0,tp_x_2,tp_t_2,compute_K('poly',tp_x_2,1.29,1))) > 1e-9).reshape(-1)] - np.dot(tp_x_2[((cvx_fiting(1.0,tp_x_2,tp_t_2,compute_K('poly',tp_x_2,1.29,1))) > 1e-9).reshape(-1)], w3)
p6+= get_scores(tp_x_1,tp_t_1,w3,b3[0])
p6+= get_scores(tp_x_3,tp_t_3,w3,b3[0])
print('Cross_validation score',p6/6)   
end6 = timer()
print('TIME',end6 - start6)
         


# In[ ]:




