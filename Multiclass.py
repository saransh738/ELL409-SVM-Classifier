#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libsvm.svmutil import *
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 

#Reading files
data_points = pd.read_csv('2019MT60763.csv', header = None, nrows = 3000)
#data = np.array((data_points.sort_values(data_points.columns[25])).values)
data = np.array(data_points.values)
dp = np.array(data)

features = data[:,:25]
class_label = dp[:,25]
tp_x, text_x, tp_t, text_t = train_test_split(features,class_label,test_size=0.2, random_state=30, stratify=class_label)

# counting no of occurence of labels of each class
unique, counts = np.unique(class_label, return_counts=True)
dict(zip(unique, counts))
#print(counts)

# for 25 features 

PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='linear'))])
parameters = {'SVM__C':np.logspace(0, 1, 10)}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x, text_t)
print ('Training score',G.score(text_x, text_t))
print ('Cross_validation score',G.score(tp_x, tp_t))
print (G.best_params_)
    
fig = plt.figure(1)
c = np.logspace(0, 1, 10)
matrix = np.zeros((10,3))
for i in range (10):
    svc = svm.SVC(kernel='linear',C = c[i])
    svc.fit(text_x, text_t)
    matrix[i][0] = i
    matrix[i][1] = svc.score(text_x, text_t)
    matrix[i][2] = svc.score(tp_x, tp_t)
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
print ('Cross_validation score',G.score(tp_x, tp_t))
print (G.best_params_)

puto = np.zeros((100,1))
luto = np.zeros((100,1))
c = np.logspace(0, 1, 10)
g = np.logspace(0, 1, 10)

for i in range (10):
    for j in range(10):
        svc = svm.SVC(kernel='rbf',C = c[i],gamma = g[j])
        svc.fit(text_x, text_t)
        puto[10*i+j][0] = svc.score(text_x, text_t)
        luto[10*i+j][0] = svc.score(tp_x, tp_t)
        
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



PIPE = Pipeline([('scaler', StandardScaler()), ('SVM', svm.SVC(kernel='poly'))])
parameters = {'SVM__C':np.logspace(0, 1, 10), 'SVM__gamma':np.logspace(0, 1, 10),'SVM__degree':[1,5]}
G = GridSearchCV(PIPE, param_grid=parameters, cv=5)

G.fit(text_x, text_t)
print ('Training score',G.score(text_x, text_t))
print ('Cross_validation score',G.score(tp_x, tp_t))
print (G.best_params_)

suto = np.zeros((100,1))
nuto = np.zeros((100,1))
c = np.logspace(0, 1, 10)
g = np.logspace(0, 1, 10)

for i in range (10):
    for j in range(10):
        svc = svm.SVC(kernel='poly',C = c[i],gamma = g[j],degree = 1)
        svc.fit(text_x, text_t)
        suto[10*i+j][0] = svc.score(text_x, text_t)
        nuto[10*i+j][0] = svc.score(tp_x, tp_t)
        
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

G.fit(text_x, text_t)
print ('Training score',G.score(text_x, text_t))
print ('Cross_validation score',G.score(tp_x, tp_t))
print (G.best_params_)



jito = np.zeros((100,1))
kito = np.zeros((100,1))
c = np.logspace(0, 1, 10)
g = np.logspace(0, 1, 10)

for i in range (10):
    for j in range(10):
        svc = svm.SVC(kernel='sigmoid',C = c[i],gamma = g[j])
        svc.fit(text_x, text_t)
        jito[10*i+j][0] = svc.score(text_x, text_t)
        kito[10*i+j][0] = svc.score(tp_x, tp_t)
        
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





# In[ ]:




