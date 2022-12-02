# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:00:40 2022

@author: campu
"""

import numpy as np
import scipy as sp
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sklearn as sk
from sklearn import datasets
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn import multiclass
from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import time

plt.close('all')

#%% Load the data

wine = sk.datasets.load_wine()
data = wine['data']
feature_names = wine['feature_names']
feature = np.asarray(data[:,(0,1,9,6)])
target = np.asarray(wine['target'],dtype='int')
combine = np.vstack((feature[:,0],feature[:,1],
                     feature[:,2],feature[:,3],target))
y_pred = []
kf = KFold(n_splits=3,shuffle=True,random_state = 112233)
kf.split(feature)

for i in range(len(target)):
    if target[i]!=0:
        if target[i]==1:
            target[i]=2
        else:
            target[i]=3
    else:
        target[i] = 1

#%% Classifier Fitting and Prediction

models = [sk.multiclass.OneVsOneClassifier(sk.linear_model.SGDClassifier()),
       sk.multiclass.OneVsOneClassifier(sk.linear_model.LogisticRegression()),
       sk.multiclass.OneVsRestClassifier(sk.linear_model.SGDClassifier()),
       sk.multiclass.OneVsRestClassifier(sk.linear_model.LogisticRegression())]
for i in models:
    for train_index, test_index in kf.split(feature):
        X_train,X_test = feature[train_index], feature[test_index]
        Y_train,Y_test = target[train_index], target[test_index]        
        i.fit(X_train, Y_train)
    y_pred.append(i.predict(X_test))
    
#%%Calculate metrics w/ SKlearn

classifiers = ['OvO SGD','OvO LogReg','OvR SGD','OvR LogReg']
ovo_ovr = ['OneVsOne','OneVsOne','OneVsRest','OneVsRest']
for i in range(len(y_pred)):
    print(classifiers[i])
    accuracy_SK = sk.metrics.accuracy_score(Y_test,y_pred[i])
    precision_SK = sk.metrics.precision_score(Y_test,y_pred[i],average=None)
    recall_SK = sk.metrics.recall_score(Y_test,y_pred[i],average=None)
    F1 = sk.metrics.f1_score(Y_test,y_pred[i],average=None) 
    print('Accuracy: '+ str(accuracy_SK))
    print('Precision: '+str(precision_SK))
    print('Recall: '+str(recall_SK))
    print('F1 Score: '+str(F1)+'\n') 

#%% Building Conf. Matrix

cm = []
for i in range(len(classifiers)):
    cm_new = confusion_matrix(Y_test,y_pred[i])
    cm_new = np.insert(cm_new,0,[0,0,0],axis=0)
    cm_new = np.insert(cm_new,0,[0,0,0,0],axis=1)
    cm.append(cm_new)
    
    
    
#%% Plotting

Y_names = ['class 1','class 2','class 3']
X=feature
Y=target
colors = mcolors.TABLEAU_COLORS
plt.rc('font',size=12)
fig, axis = plt.subplots(1,2, figsize=(7,2))
axis[0].scatter(X[Y==0,0],X[Y==0,1],marker='o',label=Y_names[0],s=15)
axis[0].scatter(X[Y==1,0],X[Y==1,1],marker='s',label=Y_names[1],s=15)
axis[0].scatter(X[Y==2,0],X[Y==2,1],marker='x',label=Y_names[2],s=15)
axis[0].set(xlabel=feature_names[0])
axis[0].set(ylabel=feature_names[1])
axis[1].scatter(X[Y==0,2],X[Y==0,3],marker='o',label=Y_names[0],s=15)
axis[1].scatter(X[Y==1,2],X[Y==1,3],marker='s',label=Y_names[1],s=15)
axis[1].scatter(X[Y==2,2],X[Y==2,3],marker='x',label=Y_names[2],s=15)
axis[1].set(xlabel=feature_names[9])
axis[1].set(ylabel=feature_names[6])
axis[1].legend()
plt.tight_layout()
plt.savefig('features',dpi=600)

fig, axis = plt.subplots(1,2, figsize=(7,3))
axis[0].scatter(X_train[Y_train==0,0],X_train[Y_train==0,1],marker='o'
                ,label=Y_names[0]+' train',s=15)
axis[0].scatter(X_train[Y_train==1,0],X_train[Y_train==1,1],marker='v'
                ,label=Y_names[1]+' train',s=15)
axis[0].scatter(X_train[Y_train==2,0],X_train[Y_train==2,1],marker='^'
                ,label=Y_names[2]+' train',s=15)
axis[0].scatter(X_test[Y_test==0,0],X_test[Y_test==0,1],marker='<'
                ,label=Y_names[0]+' test',s=15,color = colors['tab:blue'])
axis[0].scatter(X_test[Y_test==1,0],X_test[Y_test==1,1],marker='>'
                ,label=Y_names[0]+' test',s=15,color = colors['tab:orange'])
axis[0].scatter(X_test[Y_test==2,0],X_test[Y_test==2,1],marker='1'
                ,label=Y_names[0]+' test',s=15,color = colors['tab:green'])
axis[0].set(xlabel=feature_names[0])
axis[0].set(ylabel=feature_names[1])
axis[1].scatter(X_train[Y_train==0,2],X_train[Y_train==0,3],marker='o'
                ,label=Y_names[0]+' train',s=15)
axis[1].scatter(X_train[Y_train==1,2],X_train[Y_train==1,3],marker='v'
                ,label=Y_names[1]+' train',s=15)
axis[1].scatter(X_train[Y_train==2,2],X_train[Y_train==2,3],marker='^'
                ,label=Y_names[2]+' train',s=15)
axis[1].scatter(X_test[Y_test==0,2],X_test[Y_test==0,3],marker='<'
                ,label=Y_names[0]+' test',s=15,color = colors['tab:blue'])
axis[1].scatter(X_test[Y_test==1,2],X_test[Y_test==1,3],marker='>'
                ,label=Y_names[1]+' test',s=15,color = colors['tab:orange'])
axis[1].scatter(X_test[Y_test==2,2],X_test[Y_test==2,3],marker='1'
                ,label=Y_names[2]+' test',s=15,color = colors['tab:green'])
axis[1].set(xlabel=feature_names[9])
axis[1].set(ylabel=feature_names[6])
axis[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig('features',dpi=600)




fig1, axs = plt.subplots(1,4, sharey = True, sharex = True, figsize=(6,3))
plt.tight_layout(pad=.05)
fig1.suptitle(ovo_ovr[0]+ovo_ovr[2],fontsize=12)
axs[0].set_xlim([1,1])
axs[0].set_ylim([1,3])
axs[1].set_xlim([1,3])
axs[1].set_ylim([1,3])
for (j,i),label in np.ndenumerate(cm[0]):
    if label <= 7:
        axs[0].text(i,j,label,ha='center',va='center',color='white',size=8)
    else:
        axs[0].text(i,j,label,ha='center',va='center',size=8)
        
for (j,i),label in np.ndenumerate(cm[1]):
    if label <= 7:
        axs[1].text(i,j,label,ha='center',va='center',color='white',size=8)
    else:
        axs[1].text(i,j,label,ha='center',va='center',size=8)       
axs[0].imshow(cm[0],cmap='gray',aspect='equal')
axs[0].set_title('SGD')
axs[0].set_xticks([1,2,3])
axs[0].set_yticks([1,2,3])
axs[1].imshow(cm[1],cmap='gray',aspect='equal')
axs[1].set_title('Log. Reg.')
axs[1].set_xticks([1,2,3])
axs[1].set_yticks([1,2,3])
axs[0].set(ylabel='True Values')
axs[0].set(xlabel='Predicted Values')
axs[2].set_xlim([1,3])
axs[2].set_ylim([1,3])
axs[3].set_xlim([1,3])
axs[3].set_ylim([1,3])
for (j,i),label in np.ndenumerate(cm[2]):
    if label <= 7:
        axs[2].text(i,j,label,ha='center',va='center',color='white',size=8)
    else:
        axs[2].text(i,j,label,ha='center',va='center',size=8)
        
for (j,i),label in np.ndenumerate(cm[3]):
    if label <= 7:
        axs[3].text(i,j,label,ha='center',va='center',color='white',size=8)
    else:
        axs[3].text(i,j,label,ha='center',va='center',size=8)
axs[2].imshow(cm[2],cmap='gray',aspect='equal')
axs[2].set_title('SGD')
axs[2].set_xticks([1,2,3])
axs[2].set_yticks([1,2,3])
axs[3].set_xticks([1,2,3])
axs[3].set_yticks([1,2,3])
axs[3].imshow(cm[3],cmap='gray',aspect='equal')
axs[3].set_title('Log. Reg.')



plt.savefig('confusion_matrix',dpi=500)

























































































































































































































































