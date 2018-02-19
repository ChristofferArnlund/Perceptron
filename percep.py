# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:44:23 2018

@author: Christoffer
"""

import numpy as np
from matplotlib import pyplot as plt

X=np.array([
 [15694,  1042    ,-1] ,
 [18317 , 1215    ,-1],
 [25486  ,1784    ,-1],
 [29945  ,2014    ,-1],
 [30899  ,2126    ,-1],
 [36231  ,2487    ,-1],
 [36961  ,2503    ,-1],
 [37497  ,2641    ,-1],
 [37709  ,2643    ,-1],
 [40398  ,2766   , -1],
 [40588  ,2805  ,  -1],
 [43621  ,2992 ,   -1],
 [74105  ,5047,    -1],
 [75255  ,5062    ,-1],
 [76725  ,5312    ,-1],
 [15162  , 990    ,-1],
 [18031  ,1119    ,-1],
 [24843  ,1627    ,-1],
 [29800  ,1865    ,-1],
 [31030  ,1993    ,-1],
 [35298  ,2274    ,-1],
 [35680  ,2217    ,-1],
 [36172  ,2375    ,-1],
 [37464  ,2396    ,-1],
 [39552  ,2560    ,-1],
 [40255  ,2606    ,-1],
 [42514  ,2761   , -1],
 [72545  ,4597  ,  -1],
 [75352  ,4871 ,   -1],
 [74532  ,4805,    -1]])
y=np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
 -1, -1, -1, -1, -1])

for d, sample in enumerate(X):
    if y[d] < 1 :
        plt.scatter(sample[0],sample[1],s=120,marker='.',linewidths=2,color='blue')
    else:
        plt.scatter(sample[0],sample[1],s=120,marker='.',linewidths=2,color='red')
        
        
def svm_sgd_plot(X, Y):
    #init SVMs weight vector with zeros (3 vals)
    
    w = np.zeros(len(X[0]))
    #Learning rate
    eta = 1
    # #iterations
    epochs = 0
    print(epochs)
    #store missclassifications so we can plot the dt
    #change over time
    errors = []
    total_error=-1
    while total_error!=0:
        #error = 0
        total_error = 0
        epochs+=1
        for i, x in enumerate(X):
 
            # ------------------PERCEPTRON --------------
            if (np.dot(X[i], w)*Y[i]) <= 0:
                eta = eta*0.95
                total_error += (np.dot(X[i], w)*Y[i])
                w = w + eta*X[i]*Y[i]
                
                 
        errors.append(total_error*-1)

    plt.figure()
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    return w,epochs
w,epochs=svm_sgd_plot(X,y)
plt.figure()
for d, sample in enumerate(X):
    #plot the negative samples
    
    for d, sample in enumerate(X):
        if y[d] < 0 :
            plt.scatter(sample[0],sample[1],s=120,marker='.',linewidths=2,color='blue')
        else:
            plt.scatter(sample[0],sample[1],s=120,marker='.',linewidths=2,color='red')


# print the hyperplane calculated by svm_sgd()
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]
print(w[0])
print(w[1])

x2x3 = np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax=plt.gca()
ax.quiver(X,Y,U,V,scale=1,color='blue')