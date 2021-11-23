#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libsvm.svmutil import *
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
import csv

#Reading files
data_points = pd.read_csv('train_set.csv', header = None, nrows = 8000)
data = np.array(data_points.values)
dp = np.array(data)

features = data[:,:25]
class_label = dp[:,25]
text_x = (dp[0:6400,:25])
text_t = (dp[0:6400,25])

tp_x_1 = (dp[:2700,:25])
tp_t_1 = (dp[:2700,25])

tp_x_2 = (dp[2701:5400,:25])
tp_t_2 = (dp[2701:5400,25])

tp_x_3 = (dp[5401:8000,:25])
tp_t_3 = (dp[5401:8000,25])

data_points_train = pd.read_csv('test_set.csv', header = None, nrows = 2000)
dp = np.array(data_points_train.values)


# counting no of occurence of labels of each class
unique, counts = np.unique(class_label, return_counts=True)
dict(zip(unique, counts))
#print(counts)


#ploting graph varying Degree keeping other parameters default 
matrix = np.zeros((5,3))
for m in range(1,5):
    matrix[m][0] = m
    svc = svm.SVC(kernel='poly',degree = m)
    svc.fit(text_x, text_t)
    matrix[m][1] = svc.score(text_x, text_t)
    svc.fit(tp_x_1, tp_t_1)
    x = svc.score(tp_x_1, tp_t_1)
    x+= svc.score(tp_x_2, tp_t_2)
    svc.fit(tp_x_2, tp_t_2)
    x+= svc.score(tp_x_1, tp_t_1)
    x+= svc.score(tp_x_3, tp_t_3)
    svc.fit(tp_x_3, tp_t_3)
    x+= svc.score(tp_x_1, tp_t_1)
    x+= svc.score(tp_x_2, tp_t_2)
    matrix[m][2]=  x/6

fig = plt.figure(1)
plt.plot(matrix[1:,0:1],matrix[1:,1:2],label = 'Training')
plt.plot(matrix[1:,0:1],matrix[1:,2:3],label = 'Cross validation')
plt.xlabel('Degree')
plt.ylabel('Accuracy')
plt.title('Degree vs Accuracy')
plt.legend()
plt.show()



#ploting graph varying C keeping degree = 3 and other parameters default 
matrix = np.zeros((10,3))
for m in range(1,10):
    matrix[m][0] = m
    svc = svm.SVC(kernel='poly',C = m,degree = 3)
    svc.fit(text_x, text_t)
    matrix[m][1] = svc.score(text_x, text_t)
    svc.fit(tp_x_1, tp_t_1)
    x1 = svc.score(tp_x_3, tp_t_3)
    x1+= svc.score(tp_x_2, tp_t_2)
    svc.fit(tp_x_2, tp_t_2)
    x1+= svc.score(tp_x_1, tp_t_1)
    x1+= svc.score(tp_x_3, tp_t_3)
    svc.fit(tp_x_3, tp_t_3)
    x1+= svc.score(tp_x_1, tp_t_1)
    x1+= svc.score(tp_x_2, tp_t_2)
    matrix[m][2]=  x1/6

fig = plt.figure(1)
plt.plot(matrix[1:,0:1],matrix[1:,1:2],label = 'Training')
plt.plot(matrix[1:,0:1],matrix[1:,2:3],label = 'Cross validation')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('C vs Accuracy')
plt.legend()
plt.show()

#ploting graph varying gamma keeping degree = 3 and other parameters default 
matrix = np.zeros((10,3))
for m in range(1,10):
    matrix[m][0] = m
    svc = svm.SVC(kernel='poly',C = 4,degree = 3,gamma=m/100)
    svc.fit(text_x, text_t)
    matrix[m][1] = svc.score(text_x, text_t)
    svc.fit(tp_x_1, tp_t_1)
    x2 = svc.score(tp_x_3, tp_t_3)
    x2+= svc.score(tp_x_2, tp_t_2)
    svc.fit(tp_x_2, tp_t_2)
    x2+= svc.score(tp_x_1, tp_t_1)
    x2+= svc.score(tp_x_3, tp_t_3)
    svc.fit(tp_x_3, tp_t_3)
    x2+= svc.score(tp_x_1, tp_t_1)
    x2+= svc.score(tp_x_2, tp_t_2)
    matrix[m][2]=  x2/6

fig = plt.figure(1)
plt.plot(matrix[1:,0:1],matrix[1:,1:2],label = 'Training')
plt.plot(matrix[1:,0:1],matrix[1:,2:3],label = 'Cross validation')
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Gamma vs Accuracy')
plt.legend()
plt.show()

#ploting graph varying Coef keeping C = 4 , degree = 3 and other parameters default 
matrix = np.zeros((10,3))
for m in range(1,10):
    matrix[m][0] = m
    svc = svm.SVC(kernel='poly',C = 4,gamma = 0.08,degree = 3,coef0 = m/10)
    svc.fit(text_x, text_t)
    matrix[m][1] = svc.score(text_x, text_t)
    svc.fit(tp_x_1, tp_t_1)
    x3 = svc.score(tp_x_3, tp_t_3)
    x3+= svc.score(tp_x_2, tp_t_2)
    svc.fit(tp_x_2, tp_t_2)
    x3+= svc.score(tp_x_1, tp_t_1)
    x3+= svc.score(tp_x_3, tp_t_3)
    svc.fit(tp_x_3, tp_t_3)
    x3+= svc.score(tp_x_1, tp_t_1)
    x3+= svc.score(tp_x_2, tp_t_2)
    matrix[m][2]=  x3/6

fig = plt.figure(1)
plt.plot(matrix[1:,0:1],matrix[1:,1:2],label = 'Training')
plt.plot(matrix[1:,0:1],matrix[1:,2:3],label = 'Cross validation')
plt.xlabel('Coef')
plt.ylabel('Accuracy')
plt.title('Coef vs Accuracy')
plt.legend()
plt.show()

#ploting graph varying C keeping other parameters default 
matrix = np.zeros((10,3))
for m in range(1,10):
    matrix[m][0] = m
    svc = svm.SVC(kernel='rbf',C = m)
    svc.fit(text_x, text_t)
    matrix[m][1] = svc.score(text_x, text_t)
    svc.fit(tp_x_1, tp_t_1)
    x1 = svc.score(tp_x_3, tp_t_3)
    x1+= svc.score(tp_x_2, tp_t_2)
    svc.fit(tp_x_2, tp_t_2)
    x1+= svc.score(tp_x_1, tp_t_1)
    x1+= svc.score(tp_x_3, tp_t_3)
    svc.fit(tp_x_3, tp_t_3)
    x1+= svc.score(tp_x_1, tp_t_1)
    x1+= svc.score(tp_x_2, tp_t_2)
    matrix[m][2]=  x1/6

fig = plt.figure(1)
plt.plot(matrix[1:,0:1],matrix[1:,1:2],label = 'Training')
plt.plot(matrix[1:,0:1],matrix[1:,2:3],label = 'Cross validation')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('C vs Accuracy')
plt.legend()
plt.show()

 
matrix = np.zeros((10,3))
for m in range(1,10):
    matrix[m][0] = m
    svc = svm.SVC(kernel='rbf',C = 5,gamma=m/100)
    svc.fit(text_x, text_t)
    matrix[m][1] = svc.score(text_x, text_t)
    svc.fit(tp_x_1, tp_t_1)
    x2 = svc.score(tp_x_3, tp_t_3)
    x2+= svc.score(tp_x_2, tp_t_2)
    svc.fit(tp_x_2, tp_t_2)
    x2+= svc.score(tp_x_1, tp_t_1)
    x2+= svc.score(tp_x_3, tp_t_3)
    svc.fit(tp_x_3, tp_t_3)
    x2+= svc.score(tp_x_1, tp_t_1)
    x2+= svc.score(tp_x_2, tp_t_2)
    matrix[m][2]=  x2/6

fig = plt.figure(1)
plt.plot(matrix[1:,0:1],matrix[1:,1:2],label = 'Training')
plt.plot(matrix[1:,0:1],matrix[1:,2:3],label = 'Cross validation')
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Gamma vs Accuracy')
plt.legend()
plt.show()


matrix = np.zeros((10,3))
for m in range(1,10):
    matrix[m][0] = m
    svc = svm.SVC(kernel='rbf',C = 5,gamma = 0.04,coef0 = m/10)
    svc.fit(text_x, text_t)
    matrix[m][1] = svc.score(text_x, text_t)
    svc.fit(tp_x_1, tp_t_1)
    x3 = svc.score(tp_x_3, tp_t_3)
    x3+= svc.score(tp_x_2, tp_t_2)
    svc.fit(tp_x_2, tp_t_2)
    x3+= svc.score(tp_x_1, tp_t_1)
    x3+= svc.score(tp_x_3, tp_t_3)
    svc.fit(tp_x_3, tp_t_3)
    x3+= svc.score(tp_x_1, tp_t_1)
    x3+= svc.score(tp_x_2, tp_t_2)
    matrix[m][2]=  x3/6

fig = plt.figure(1)
plt.plot(matrix[1:,0:1],matrix[1:,1:2],label = 'Training')
plt.plot(matrix[1:,0:1],matrix[1:,2:3],label = 'Cross validation')
plt.xlabel('Coef')
plt.ylabel('Accuracy')
plt.title('Coef vs Accuracy')
plt.legend()
plt.show()


svc = svm.SVC(kernel='rbf',C = 5,gamma = 0.04,coef0 = 0.410)
svc.fit(text_x, text_t)
print('Training score',svc.score(text_x, text_t))
svc.fit(tp_x_1,tp_t_1)
x8 = svc.score(tp_x_2, tp_t_2)
x8+=svc.score(tp_x_3, tp_t_3)
svc.fit(tp_x_2,tp_t_2)
x8+=svc.score(tp_x_3, tp_t_3)
x8+= svc.score(tp_x_1, tp_t_1)
svc.fit(tp_x_3,tp_t_3)
x8+=svc.score(tp_x_1, tp_t_1)
x8+=svc.score(tp_x_2, tp_t_2)
print('Cross_validation score',x8/6)

with open('prediction.csv', 'w+') as f:
    f.write('Id,Class\n')
    for i in range(len(svc.predict(dp))):
        if i + 1 < 1000: 
            f.write('{},{:d}\n'.format(str(i+1), int(svc.predict(dp)[i])))
        else:
            f.write('\"{:01d},{:03d}\",{:d}\n'.format((i+1)//1000, (i+1)%1000, int(svc.predict(dp)[i])))


# In[ ]:




