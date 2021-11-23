#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libsvm.svmutil import *
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from timeit import default_timer as timer

#Reading files
data_points_train = pd.read_csv('2019MT60763.csv', header = None, nrows = 3000)
data = np.array((data_points_train.sort_values(data_points_train.columns[25])).values)
dp = np.array(data)

class_label = dp[:,25]

# counting no of occurence of labels of each class
unique, counts = np.unique(class_label, return_counts=True)
dict(zip(unique, counts))
#print(counts)


# for 25 features 

# FOR CLASSES {0,1}

text_x = dp[:631,:25]
text_t = dp[:631,25]

# for cross_validation 
tp_x_1 = np.append(dp[:100,:25],dp[306:406,:25],axis=0)
tp_t_1 = np.append(dp[:100,25],dp[306:406,25],axis=0)

tp_x_2 = np.append(dp[101:201,:25],dp[407:507,:25],axis=0)
tp_t_2 = np.append(dp[101:201,25],dp[407:507,25],axis=0)

tp_x_3 = np.append(dp[202:305,:25],dp[508:631,:25],axis=0)
tp_t_3 = np.append(dp[202:305,25],dp[508:631,25],axis=0)



PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='linear'))])
parameters = {'SVM__C':np.logspace(0, 1, 10)}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x, text_t)
print ('Training score',G.score(text_x, text_t))
print (G.best_params_)
G.fit(tp_x_1,tp_t_1)
x = G.score(tp_x_2, tp_t_2)
x+=G.score(tp_x_3, tp_t_3)
G.fit(tp_x_2,tp_t_2)
x+=G.score(tp_x_3, tp_t_3)
x+=G.score(tp_x_1, tp_t_1)
G.fit(tp_x_3,tp_t_3)
x+=G.score(tp_x_2, tp_t_2)
x+=G.score(tp_x_1, tp_t_1)
print('Cross_validation score',x/6)
print(((svm.SVC(kernel = 'linear', C = 1)).fit(text_x,text_t)).support_)


fig = plt.figure(1)
c = np.logspace(0, 1, 10)
matrix = np.zeros((10,3))
for i in range (10):
    svc = svm.SVC(kernel='linear',C = c[i])
    svc.fit(text_x, text_t)
    matrix[i][0] = i
    matrix[i][1] = svc.score(text_x, text_t)
    svc.fit(tp_x_1,tp_t_1)
    x1 = svc.score(tp_x_2, tp_t_2)
    x1+=svc.score(tp_x_3, tp_t_3)
    svc.fit(tp_x_2,tp_t_2)
    x1+=svc.score(tp_x_3, tp_t_3)
    x1+=svc.score(tp_x_1, tp_t_1)
    svc.fit(tp_x_3,tp_t_3)
    x1+=svc.score(tp_x_2, tp_t_2)
    x1+=svc.score(tp_x_1, tp_t_1)
    matrix[i][2] = x1/6
plt.plot(matrix[:,0:1],matrix[:,1:2],label = 'cross_validation score')
plt.plot(matrix[:,0:1],matrix[:,2:3],label = 'Training score')
plt.title('C vs Accuracy')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.legend()
plt.show()


PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='rbf'))])
parameters = {'SVM__C':np.logspace(0, 1, 10), 'SVM__gamma':np.logspace(0, 1, 10)}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x, text_t)
print ('Training score',G.score(text_x, text_t))
print (G.best_params_)
G.fit(tp_x_1,tp_t_1)
y = G.score(tp_x_2, tp_t_2)
y+=G.score(tp_x_3, tp_t_3)
G.fit(tp_x_2,tp_t_2)
y+=G.score(tp_x_3, tp_t_3)
y+=G.score(tp_x_1, tp_t_1)
G.fit(tp_x_3,tp_t_3)
y+=G.score(tp_x_2, tp_t_2)
y+=G.score(tp_x_1, tp_t_1)
print('Cross_validation score',y/6)
print(((svm.SVC(kernel = 'rbf', C = 1.29,gamma = 1)).fit(text_x,text_t)).support_)

puto = np.zeros((100,1))
luto = np.zeros((100,1))
c = np.logspace(0, 1, 10)
g = np.logspace(0, 1, 10)

for i in range (10):
    for j in range(10):
        svc = svm.SVC(kernel='rbf',C = c[i],gamma = g[j])
        svc.fit(text_x, text_t)
        puto[10*i+j][0] = svc.score(text_x, text_t)
        svc.fit(tp_x_1,tp_t_1)
        y1 = svc.score(tp_x_2, tp_t_2)
        y1+=svc.score(tp_x_3, tp_t_3)
        svc.fit(tp_x_2,tp_t_2)
        y1+=svc.score(tp_x_3, tp_t_3)
        y1+=svc.score(tp_x_1, tp_t_1)
        svc.fit(tp_x_3,tp_t_3)
        y1+=svc.score(tp_x_2, tp_t_2)
        y1+=svc.score(tp_x_1, tp_t_1)
        luto[10*i+j][0] = y1/6
        
g, c = np.meshgrid(g, c)
graph = np.ravel(puto)
patrix = np.ravel(luto)
patrix = patrix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, patrix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (cross-validation)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()

graph = graph.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, graph)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (training)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()

        
start = timer()
PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='poly'))])
parameters = {'SVM__C':np.logspace(0, 1, 10), 'SVM__gamma':np.logspace(0, 1, 10),'SVM__degree':[1,5]}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x, text_t)
print ('Training score',G.score(text_x, text_t))
print (G.best_params_)
G.fit(tp_x_1,tp_t_1)
z = G.score(tp_x_2, tp_t_2)
z+=G.score(tp_x_3, tp_t_3)
G.fit(tp_x_2,tp_t_2)
z+=G.score(tp_x_3, tp_t_3)
z+=G.score(tp_x_1, tp_t_1)
G.fit(tp_x_3,tp_t_3)
z+=G.score(tp_x_2, tp_t_2)
z+=G.score(tp_x_1, tp_t_1)
print('Cross_validation score',z/6)
end = timer()
print('TIME',end - start) 
print(((svm.SVC(kernel = 'poly', C = 1,gamma = 1,degree = 1)).fit(text_x,text_t)).support_)

suto = np.zeros((100,1))
nuto = np.zeros((100,1))
c = np.logspace(0, 1, 10)
g = np.logspace(0, 1, 10)

for i in range (10):
    for j in range(10):
        svc = svm.SVC(kernel='poly',C = c[i],gamma = g[j],degree = 1)
        svc.fit(text_x, text_t)
        suto[10*i+j][0] = svc.score(text_x, text_t)
        svc.fit(tp_x_1,tp_t_1)
        z1 = svc.score(tp_x_2, tp_t_2)
        z1+=svc.score(tp_x_3, tp_t_3)
        svc.fit(tp_x_2,tp_t_2)
        z1+=svc.score(tp_x_3, tp_t_3)
        z1+=svc.score(tp_x_1, tp_t_1)
        svc.fit(tp_x_3,tp_t_3)
        z1+=svc.score(tp_x_2, tp_t_2)
        z1+=svc.score(tp_x_1, tp_t_1)
        nuto[10*i+j][0] = z1/6
        
g, c = np.meshgrid(g, c)
trix = np.ravel(suto)
prix = np.ravel(nuto)
prix = prix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, prix)
cbar = fig.colorbar(k)
plt.xlabel('C')
plt.ylabel('gamma')
plt.title('Contour plot for Accuracy v/s C and gamma (cross-validation)')
plt.xscale('log')
plt.yscale('log')
plt.show()
# training
trix = trix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, trix)
cbar = fig.colorbar(k)
plt.xlabel('C')
plt.ylabel('gamma')
plt.title('Contour plot for Accuracy v/s C and gamma (training)')
plt.xscale('log')
plt.yscale('log')
plt.show()


PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='sigmoid'))])
parameters = {'SVM__C':np.logspace(0, 1, 10), 'SVM__gamma':np.logspace(0, 1, 10)}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x, text_t)
print ('Training score',G.score(text_x, text_t))
print (G.best_params_)
G.fit(tp_x_1,tp_t_1)
f = G.score(tp_x_2, tp_t_2)
f+=G.score(tp_x_3, tp_t_3)
G.fit(tp_x_2,tp_t_2)
f+=G.score(tp_x_3, tp_t_3)
f+=G.score(tp_x_1, tp_t_1)
G.fit(tp_x_3,tp_t_3)
f+=G.score(tp_x_2, tp_t_2)
f+=G.score(tp_x_1, tp_t_1)
print('Cross_validation score',f/6)
print(((svm.SVC(kernel = 'sigmoid', C = 10,gamma = 1)).fit(text_x,text_t)).support_)



jito = np.zeros((100,1))
kito = np.zeros((100,1))
c = np.logspace(0, 1, 10)
g = np.logspace(0, 1, 10)

for i in range (10):
    for j in range(10):
        svc = svm.SVC(kernel='sigmoid',C = c[i],gamma = g[j])
        svc.fit(text_x, text_t)
        jito[10*i+j][0] = svc.score(text_x, text_t)
        svc.fit(tp_x_1,tp_t_1)
        f1 = svc.score(tp_x_2, tp_t_2)
        f1+=svc.score(tp_x_3, tp_t_3)
        svc.fit(tp_x_2,tp_t_2)
        f1+=svc.score(tp_x_3, tp_t_3)
        f1+=svc.score(tp_x_1, tp_t_1)
        svc.fit(tp_x_3,tp_t_3)
        f1+=svc.score(tp_x_2, tp_t_2)
        f1+=svc.score(tp_x_1, tp_t_1)
        kito[10*i+j][0] = f1/6
        
g, c = np.meshgrid(g, c)
tatrix = np.ravel(jito)
katrix = np.ravel(kito)
katrix = katrix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, katrix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (cross-validation)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()

tatrix = tatrix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, tatrix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (training)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()


# In[5]:


# FOR CLASSES {2,3}


text_x_2 = (dp[632:1230,:25])
text_t_2 = (dp[632:1230,25])


# for cross_validation 
tp_x_1 = np.append(dp[632:732,:25],dp[943:1043,:25],axis=0)
tp_t_1 = np.append(dp[632:732,25],dp[943:1043,25],axis=0)

tp_x_2 = np.append(dp[732:832,:25],dp[1043:1143,:25],axis=0)
tp_t_2 = np.append(dp[732:832,25],dp[1043:1143,25],axis=0)

tp_x_3 = np.append(dp[832:942,:25],dp[1143:1230,:25],axis=0)
tp_t_3 = np.append(dp[832:942,25],dp[1143:1230,25],axis=0)


PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='linear'))])
parameters = {'SVM__C':np.logspace(0, 1, 10)}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x_2, text_t_2)
print ('Training score',G.score(text_x_2, text_t_2))
print (G.best_params_)
G.fit(tp_x_1,tp_t_1)
l1 = G.score(tp_x_2, tp_t_2)
l1+=G.score(tp_x_3, tp_t_3)
G.fit(tp_x_2,tp_t_2)
l1+=G.score(tp_x_3, tp_t_3)
l1+=G.score(tp_x_1, tp_t_1)
G.fit(tp_x_3,tp_t_3)
l1+=G.score(tp_x_2, tp_t_2)
l1+=G.score(tp_x_1, tp_t_1)
print('Cross_validation score',l1/6)
print(((svm.SVC(kernel = 'linear', C = 7.74)).fit(text_x_2,text_t_2)).support_)    


fig = plt.figure(2)
c = np.logspace(0, 1, 10)
matrix = np.zeros((10,3))
for i in range (10):
    svc = svm.SVC(kernel='linear',C = c[i])
    svc.fit(text_x_2, text_t_2)
    matrix[i][0] = i
    matrix[i][1] = svc.score(text_x_2, text_t_2)
    svc.fit(tp_x_1,tp_t_1)
    l2 = svc.score(tp_x_2, tp_t_2)
    l2+=svc.score(tp_x_3, tp_t_3)
    svc.fit(tp_x_2,tp_t_2)
    l2+=svc.score(tp_x_3, tp_t_3)
    l2+=svc.score(tp_x_1, tp_t_1)
    svc.fit(tp_x_3,tp_t_3)
    l2+=svc.score(tp_x_2, tp_t_2)
    l2+=svc.score(tp_x_1, tp_t_1)
    matrix[i][2] = l2/6
plt.plot(matrix[:,0:1],matrix[:,1:2],label = 'cross_validation score')
plt.plot(matrix[:,0:1],matrix[:,2:3],label = 'Training score')
plt.title('C vs Accuracy')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.legend()
plt.show()    

PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='rbf'))])
parameters = {'SVM__C':np.logspace(0, 1, 10), 'SVM__gamma':np.logspace(0, 1, 10)}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x_2, text_t_2)
print ('Training score',G.score(text_x_2, text_t_2))
print (G.best_params_)
G.fit(tp_x_1,tp_t_1)
l3 = G.score(tp_x_2, tp_t_2)
l3+=G.score(tp_x_3, tp_t_3)
G.fit(tp_x_2,tp_t_2)
l3+=G.score(tp_x_3, tp_t_3)
l3+=G.score(tp_x_1, tp_t_1)
G.fit(tp_x_3,tp_t_3)
l3+=G.score(tp_x_2, tp_t_2)
l3+=G.score(tp_x_1, tp_t_1)
print('Cross_validation score',l3/6)
print(((svm.SVC(kernel = 'rbf', C = 1.29,gamma =1)).fit(text_x_2,text_t_2)).support_)

puto = np.zeros((100,1))
luto = np.zeros((100,1))
c = np.logspace(0, 1, 10)
g = np.logspace(0, 1, 10)

for i in range (10):
    for j in range(10):
        svc = svm.SVC(kernel='rbf',C = c[i],gamma = g[j])
        svc.fit(text_x_2, text_t_2)
        puto[10*i+j][0] = svc.score(text_x_2, text_t_2)
        svc.fit(tp_x_1,tp_t_1)
        l4 = svc.score(tp_x_2, tp_t_2)
        l4+=svc.score(tp_x_3, tp_t_3)
        svc.fit(tp_x_2,tp_t_2)
        l4+=svc.score(tp_x_3, tp_t_3)
        l4+=svc.score(tp_x_1, tp_t_1)
        svc.fit(tp_x_3,tp_t_3)
        l4+=svc.score(tp_x_2, tp_t_2)
        l4+=svc.score(tp_x_1, tp_t_1)
        luto[10*i+j][0] = l4/6
        
g, c = np.meshgrid(g, c)
graph = np.ravel(puto)
patrix = np.ravel(luto)
patrix = patrix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, patrix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (cross-validation)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()

graph = graph.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, graph)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (training)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()

start1 = timer()
PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='poly'))])
parameters = {'SVM__C':np.logspace(0, 1, 10), 'SVM__gamma':np.logspace(0, 1, 10),'SVM__degree':[1,5]}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x_2, text_t_2)
print ('Training score',G.score(text_x_2, text_t_2))
print (G.best_params_)
G.fit(tp_x_1,tp_t_1)
l5 = G.score(tp_x_2, tp_t_2)
l5+=G.score(tp_x_3, tp_t_3)
G.fit(tp_x_2,tp_t_2)
l5+=G.score(tp_x_3, tp_t_3)
l5+=G.score(tp_x_1, tp_t_1)
G.fit(tp_x_3,tp_t_3)
l5+=G.score(tp_x_2, tp_t_2)
l5+=G.score(tp_x_1, tp_t_1)
print('Cross_validation score',l5/6)
end1 = timer()
print('TIME',end1 - start1)
print(((svm.SVC(kernel = 'poly', C = 1,gamma =1 ,degree=5)).fit(text_x_2,text_t_2)).support_)

suto = np.zeros((100,1))
nuto = np.zeros((100,1))
c = np.logspace(0, 1, 10)
g = np.logspace(0, 1, 10)

for i in range (10):
    for j in range(10):
        svc = svm.SVC(kernel='poly',C = c[i],gamma = g[j],degree = 5)
        svc.fit(text_x_2, text_t_2)
        suto[10*i+j][0] = svc.score(text_x_2, text_t_2)
        svc.fit(tp_x_1,tp_t_1)
        l6 = svc.score(tp_x_2, tp_t_2)
        l6+=svc.score(tp_x_3, tp_t_3)
        svc.fit(tp_x_2,tp_t_2)
        l6+=svc.score(tp_x_3, tp_t_3)
        l6+=svc.score(tp_x_1, tp_t_1)
        svc.fit(tp_x_3,tp_t_3)
        l6+=svc.score(tp_x_2, tp_t_2)
        l6+=svc.score(tp_x_1, tp_t_1)
        nuto[10*i+j][0] = l6/6
        
g, c = np.meshgrid(g, c)
trix = np.ravel(suto)
prix = np.ravel(nuto)
prix = prix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, prix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (cross-validation)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()

trix = trix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, trix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (training)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()



PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='sigmoid'))])
parameters = {'SVM__C':np.logspace(0, 1, 10), 'SVM__gamma':np.logspace(0, 1, 10)}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x_2, text_t_2)
print ('Training score',G.score(text_x_2, text_t_2))
print (G.best_params_)
G.fit(tp_x_1,tp_t_1)
l7 = G.score(tp_x_2, tp_t_2)
l7+=G.score(tp_x_3, tp_t_3)
G.fit(tp_x_2,tp_t_2)
l7+=G.score(tp_x_3, tp_t_3)
l7+=G.score(tp_x_1, tp_t_1)
G.fit(tp_x_3,tp_t_3)
l7+=G.score(tp_x_2, tp_t_2)
l7+=G.score(tp_x_1, tp_t_1)
print('Cross_validation score',l7/6)
print(((svm.SVC(kernel = 'sigmoid', C = 1.66,gamma = 3.59 )).fit(text_x_2,text_t_2)).support_)

jito = np.zeros((100,1))
kito = np.zeros((100,1))
c = np.logspace(0, 1, 10)
g = np.logspace(0, 1, 10)

for i in range (10):
    for j in range(10):
        svc = svm.SVC(kernel='sigmoid',C = c[i],gamma = g[j])
        svc.fit(text_x_2, text_t_2)
        jito[10*i+j][0] = svc.score(text_x_2, text_t_2)
        svc.fit(tp_x_1,tp_t_1)
        l8 = svc.score(tp_x_2, tp_t_2)
        l8+=svc.score(tp_x_3, tp_t_3)
        svc.fit(tp_x_2,tp_t_2)
        l8+=svc.score(tp_x_3, tp_t_3)
        l8+=svc.score(tp_x_1, tp_t_1)
        svc.fit(tp_x_3,tp_t_3)
        l8+=svc.score(tp_x_2, tp_t_2)
        l8+=svc.score(tp_x_1, tp_t_1)
        kito[10*i+j][0] = l8/6
        
g, c = np.meshgrid(g, c)
tatrix = np.ravel(jito)
katrix = np.ravel(kito)
katrix = katrix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, katrix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (cross-validation)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()

tatrix = tatrix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, tatrix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (training)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()


# In[6]:


# FOR CLASSES {4,5}


text_x_3 = dp[1232:1800,:25]
text_t_3 = dp[1232:1800,25]


# for cross_validation 
tp_x_1 = np.append(dp[1232:1332,:25],dp[1533:1610,:25],axis=0)
tp_t_1 = np.append(dp[1232:1332,25],dp[1533:1610,25],axis=0)

tp_x_2 = np.append(dp[1333:1433,:25],dp[1610:1699,:25],axis=0)
tp_t_2 = np.append(dp[1333:1433,25],dp[1610:1699,25],axis=0)

tp_x_3 = np.append(dp[1433:1532,:25],dp[1700:1800,:25],axis=0)
tp_t_3 = np.append(dp[1433:1532,25],dp[1700:1800,25],axis=0)


PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='linear'))])
parameters = {'SVM__C':np.logspace(0, 1, 10)}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x_3, text_t_3)
print ('Training score',G.score(text_x_3, text_t_3))
print (G.best_params_)
G.fit(tp_x_1,tp_t_1)
p1 = G.score(tp_x_2, tp_t_2)
p1+=G.score(tp_x_3, tp_t_3)
G.fit(tp_x_2,tp_t_2)
p1+=G.score(tp_x_3, tp_t_3)
p1+=G.score(tp_x_1, tp_t_1)
G.fit(tp_x_3,tp_t_3)
p1+=G.score(tp_x_2, tp_t_2)
p1+=G.score(tp_x_1, tp_t_1)
print('Cross_validation score',p1/6)
print(((svm.SVC(kernel = 'linear', C = 1.29)).fit(text_x_3,text_t_3)).support_)    

fig = plt.figure(3)
c = np.logspace(0, 1, 10)
matrix = np.zeros((10,3))
for i in range (10):
    svc = svm.SVC(kernel='linear',C = c[i])
    svc.fit(text_x_3, text_t_3)
    matrix[i][0] = i
    matrix[i][1] = svc.score(text_x_3, text_t_3)
    svc.fit(tp_x_1,tp_t_1)
    p2 = svc.score(tp_x_2, tp_t_2)
    p2+=svc.score(tp_x_3, tp_t_3)
    svc.fit(tp_x_2,tp_t_2)
    p2+=svc.score(tp_x_3, tp_t_3)
    p2+=svc.score(tp_x_1, tp_t_1)
    svc.fit(tp_x_3,tp_t_3)
    p2+=svc.score(tp_x_1, tp_t_1)
    p2+=svc.score(tp_x_2, tp_t_2)
    matrix[i][2] = p2/6
plt.plot(matrix[:,0:1],matrix[:,1:2],label = 'cross_validation score')
plt.plot(matrix[:,0:1],matrix[:,2:3],label = 'Training score')
plt.title('C vs Accuracy')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.legend()
plt.show()

PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='rbf'))])
parameters = {'SVM__C':np.logspace(0, 1, 10), 'SVM__gamma':np.logspace(0, 1, 10)}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x_3, text_t_3)
print ('Training score',G.score(text_x_3, text_t_3))
print (G.best_params_)
G.fit(tp_x_1,tp_t_1)
p3 = G.score(tp_x_2, tp_t_2)
p3+=G.score(tp_x_3, tp_t_3)
G.fit(tp_x_2,tp_t_2)
p3+=G.score(tp_x_3, tp_t_3)
p3+=G.score(tp_x_1, tp_t_1)
G.fit(tp_x_3,tp_t_3)
p3+=G.score(tp_x_2, tp_t_2)
p3+=G.score(tp_x_1, tp_t_1)
print('Cross_validation score',p3/6)
print(((svm.SVC(kernel = 'rbf', C = 1.29,gamma =1)).fit(text_x_3,text_t_3)).support_)

puto = np.zeros((100,1))
luto = np.zeros((100,1))
c = np.logspace(0, 1, 10)
g = np.logspace(0, 1, 10)

for i in range (10):
    for j in range(10):
        svc = svm.SVC(kernel='rbf',C = c[i],gamma = g[j])
        svc.fit(text_x_3, text_t_3)
        puto[10*i+j][0] = svc.score(text_x_3, text_t_3)
        svc.fit(tp_x_1,tp_t_1)
        p4 = svc.score(tp_x_2, tp_t_2)
        p4+=svc.score(tp_x_3, tp_t_3)
        svc.fit(tp_x_2,tp_t_2)
        p4+=svc.score(tp_x_3, tp_t_3)
        p4+=svc.score(tp_x_1, tp_t_1)
        svc.fit(tp_x_3,tp_t_3)
        p4+=svc.score(tp_x_2, tp_t_2)
        p4+=svc.score(tp_x_1, tp_t_1)
        luto[10*i+j][0] = p4/6
        
g, c = np.meshgrid(g, c)
graph = np.ravel(puto)
patrix = np.ravel(luto)
patrix = patrix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, patrix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (cross-validation)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()

graph = graph.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, graph)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (training)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()

start2 = timer()
PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='poly'))])
parameters = {'SVM__C':np.logspace(0, 1, 10), 'SVM__gamma':np.logspace(0, 1, 10),'SVM__degree':[1,5]}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x_3, text_t_3)
print ('Training score',G.score(text_x_3, text_t_3))
print (G.best_params_)
G.fit(tp_x_1,tp_t_1)
p5 = G.score(tp_x_2, tp_t_2)
p5+=G.score(tp_x_3, tp_t_3)
G.fit(tp_x_2,tp_t_2)
p5+=G.score(tp_x_3, tp_t_3)
p5+=G.score(tp_x_1, tp_t_1)
G.fit(tp_x_3,tp_t_3)
p5+=G.score(tp_x_2, tp_t_2)
p5+=G.score(tp_x_1, tp_t_1)
print('Cross_validation score',p5/6)
end2 = timer()
print('TIME',end2 - start2) 
print(((svm.SVC(kernel = 'poly', C = 1,gamma = 1.29,degree=1)).fit(text_x_3,text_t_3)).support_)

suto = np.zeros((100,1))
nuto = np.zeros((100,1))
c = np.logspace(0, 1, 10)
g = np.logspace(0, 1, 10)

for i in range (10):
    for j in range(10):
        svc = svm.SVC(kernel='poly',C = c[i],gamma = g[j],degree = 1)
        svc.fit(text_x_3, text_t_3)
        suto[10*i+j][0] = svc.score(text_x_3, text_t_3)
        svc.fit(tp_x_1,tp_t_1)
        p8 = svc.score(tp_x_2, tp_t_2)
        p8+=svc.score(tp_x_3, tp_t_3)
        svc.fit(tp_x_2,tp_t_2)
        p8+=svc.score(tp_x_3, tp_t_3)
        p8+=svc.score(tp_x_1, tp_t_1)
        svc.fit(tp_x_3,tp_t_3)
        p8+=svc.score(tp_x_2, tp_t_2)
        p8+=svc.score(tp_x_1, tp_t_1)
        nuto[10*i+j][0] = p8/6
        
g, c = np.meshgrid(g, c)
trix = np.ravel(suto)
prix = np.ravel(nuto)
prix = prix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, prix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (cross-validation)')
plt.xlabel('C')
plt.ylabel('gamma')

plt.xscale('log')
plt.yscale('log')
plt.show()

trix = trix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, trix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (training)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()



PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='sigmoid'))])
parameters = {'SVM__C':np.logspace(0, 1, 10), 'SVM__gamma':np.logspace(0, 1, 10)}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x_3, text_t_3)
print ('Training score',G.score(text_x_3, text_t_3))
print (G.best_params_)
G.fit(tp_x_1,tp_t_1)
p6 = G.score(tp_x_2, tp_t_2)
p6+=G.score(tp_x_3, tp_t_3)
G.fit(tp_x_2,tp_t_2)
p6+=G.score(tp_x_3, tp_t_3)
p6+=G.score(tp_x_1, tp_t_1)
G.fit(tp_x_3,tp_t_3)
p6+=G.score(tp_x_2, tp_t_2)
p6+=G.score(tp_x_1, tp_t_1)
print('Cross_validation score',p6/6)
print(((svm.SVC(kernel = 'sigmoid', C = 4.64,gamma = 3.59)).fit(text_x_3,text_t_3)).support_)

ito = np.zeros((100,1))
kito = np.zeros((100,1))
c = np.logspace(0, 1, 10)
g = np.logspace(0, 1, 10)

for i in range (10):
    for j in range(10):
        svc = svm.SVC(kernel='sigmoid',C = c[i],gamma = g[j])
        svc.fit(text_x_3, text_t_3)
        jito[10*i+j][0] = svc.score(text_x_3, text_t_3)
        svc.fit(tp_x_1,tp_t_1)
        p7 = svc.score(tp_x_2, tp_t_2)
        p7+=svc.score(tp_x_3, tp_t_3)
        svc.fit(tp_x_2,tp_t_2)
        p7+=svc.score(tp_x_3, tp_t_3)
        p7+=svc.score(tp_x_1, tp_t_1)
        svc.fit(tp_x_3,tp_t_3)
        p7+=svc.score(tp_x_2, tp_t_2)
        p7+=svc.score(tp_x_1, tp_t_1)
        kito[10*i+j][0] = p7/6
        
g, c = np.meshgrid(g, c)
tatrix = np.ravel(jito)
katrix = np.ravel(kito)
katrix = katrix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, katrix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (cross-validation)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()

tatrix = tatrix.reshape(c.shape)
fig, p = plt.subplots()
k = p.contourf(c, g, tatrix)
cbar = fig.colorbar(k)
plt.title('Accuracy v/s C and gamma (training)')
plt.xlabel('C')
plt.ylabel('gamma')
plt.xscale('log')
plt.yscale('log')
plt.show()


# In[ ]:




