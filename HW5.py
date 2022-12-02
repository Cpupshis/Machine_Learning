# -*- coding: utf-8 -*-
"""
Created on Tue Nov 1 4:14:34 2022

@author: campu
"""

from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
import sklearn as sk
from sklearn import svm, datasets
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import multiclass
import time as time
from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
   
plt.close('all')

#load data
iris = sk.datasets.load_iris()
X = iris['data']
Y = iris['target']
feature_names = iris['feature_names']
target_names = iris['target_names']
Y_names = iris['target_names']
h = 0.2




# plot the sepad data
plt.figure(figsize=(6.5,3))
plt.subplot(121)
plt.grid(True)
plt.scatter(X[Y==0,0], X[Y==0,1], marker ='o', label = Y_names[0])
plt.scatter(X[Y==1,0], X[Y==1,1], marker ='s', label = Y_names[1])
plt.scatter(X[Y==2,0], X[Y==2,1], marker ='d', label = Y_names[2])
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend(framealpha=1,loc=1)

plt.subplot(122)
plt.grid(True)
plt.scatter(X[Y==0,2], X[Y==0,3], marker ='o', label = Y_names[0])
plt.scatter(X[Y==1,2], X[Y==1,3], marker ='s', label = Y_names[1])
plt.scatter(X[Y==2,2], X[Y==2,3], marker ='d', label = Y_names[2])
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.tight_layout()

# build the training and target set
X_train = X[:,(0,1)]
feature = X_train
y_train = Y

#%% Classifier Fitting and Prediction
y_pred_ovr = []
y_pred_soft = []
y_pred_softmax = np.zeros(50)
kf = KFold(n_splits=3,shuffle=True,random_state = 112233)
kf.split(feature)
C = 10

#One vs Rest
o_vs_r = OneVsRestClassifier(SVC(C=C))

# softmax regression
softmax_reg=sk.linear_model.LogisticRegression(multi_class="multinomial", 
                                               solver="lbfgs",C=C)

models = [o_vs_r,softmax_reg]
for i in models:
    for train_index, test_index in kf.split(feature):
        X_train,X_test = feature[train_index], feature[test_index]
        Y_train,Y_test = Y[train_index], Y[test_index]        
        i.fit(X_train, Y_train)
    if i != softmax_reg:  
        y_pred_ovr=models[0].predict(X_test)
    else:
        y_pred_soft=models[1].predict_proba(X_test)

for i in range(len(y_pred_soft)):
    if y_pred_soft[i,0] < .5 and y_pred_soft[i,1] > .5:
        y_pred_softmax[i] = 1
    elif y_pred_soft[i,0] < .5 and y_pred_soft[i,2] > .5:
        y_pred_softmax[i] = 2
    elif y_pred_soft[i,0] > y_pred_soft[i,1] and y_pred_soft[i,0] > y_pred_soft[i,2]:
        y_pred_softmax[i] = 0

y_pred = np.asarray([y_pred_ovr,y_pred_softmax])
    
#build the x values for the predictions over the entire pedal space
x_grid, y_grid = np.meshgrid(
    np.linspace(0,9,500), np.linspace(0,5,200)
    )
X_new = np.vstack((x_grid.reshape(-1), y_grid.reshape(-1))).T


yhat = o_vs_r.predict(X_new)

x_min, x_max = X_new[:, 0].min() - 1, X_new[:, 0].max() + 1
y_min, y_max = X_new[:, 1].min() - 1, X_new[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# predict on the vectorized format
y_predict = softmax_reg.predict(X_new)
y_proba = softmax_reg.predict_proba(np.c_[x_grid.ravel(), y_grid.ravel()])
for i in range(np.shape(y_proba)[1]):
    y_proba[:,i] *= i+1

# convert back to meshgrid shape for plotting
zz_predict = y_predict.reshape(x_grid.shape)
zz_proba = y_proba.reshape(200,500,3)
zz_proba1 = zz_proba[:,:,0]
zz_proba2 = zz_proba[:,:,1]
zz_proba3 = zz_proba[:,:,2]

for (i,j),k in np.ndenumerate(zz_proba1):
    if k <  zz_proba2[i,j]-1:
        zz_proba1[i,j] = zz_proba2[i,j]
    elif k < zz_proba3[i,j]-2:
        zz_proba1[i,j] = zz_proba3[i,j]
    else:
        zz_proba1[i,j] = k
    
#OVR?
zz_yhat = yhat.reshape(x_grid.shape)

Z = o_vs_r.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
Z = Z.reshape(x_grid.shape)

xax,yax = x_grid[0,:], y_grid[:,0]
yax = np.reshape(yax,-1)


    
#%% Plotting

fmt = {}
strs = [Y_names[0],Y_names[2]]


#plot one vs rest
fig,axs = plt.subplots(1,2,figsize=(6.5,3),sharex=True,sharey=True)

#OVR contour plot
axs[0].contourf(xax, yax, Z,2,levels = 2, cmap='Pastel2')
contour = axs[0].contour(xax, yax, Z, levels = [0,1,2],cmap=plt.cm.brg,extend='both')
for l, s in zip(contour.levels, strs):
    fmt[l] = s
axs[0].clabel(contour, contour.levels, fmt=fmt, inline=True, fontsize=12,manual=[(3.08,2.2),(6,2.1)])
axs[0].set(xlabel=feature_names[0],ylabel=feature_names[1])
axs[0].axis([3,8.25,1.5,5])
axs[0].scatter(X[Y==0,0],X[Y==0,1],marker='o',label=Y_names[0],s=15)
axs[0].scatter(X[Y==1,0],X[Y==1,1],marker='s',label=Y_names[1],s=15)
axs[0].scatter(X[Y==2,0],X[Y==2,1],marker='d',label=Y_names[2],s=15)
axs[0].set(xlabel=feature_names[0],ylabel = feature_names[1])
axs[0].legend(framealpha=1,loc=1)
axs[0].set_title('OvR $C=$'+str(C))

# plot the softmax regression
#Softmax contour plot

fmt1 = {}
strs1 = [Y_names[0],Y_names[2]]

axs[1].contourf(x_grid, y_grid, zz_proba1,levels=2, cmap='Pastel2')
contour1=axs[1].contour(x_grid, y_grid, zz_proba1,levels=[1,2,3], cmap=plt.cm.brg)
for ls, sl in zip(contour1.levels, strs):
    fmt[ls] = sl
axs[1].clabel(contour1, contour1.levels, fmt=fmt, inline=True, fontsize=12,manual=[(4,2.35),(6.75,4.35)])
axs[1].set(xlabel=feature_names[0],ylabel=feature_names[1])
axs[1].axis([3,8.25,1.5,5])
axs[1].scatter(X[Y==0,0],X[Y==0,1],marker='o',label=Y_names[0],zorder=10,s=15)
axs[1].scatter(X[Y==1,0],X[Y==1,1],marker='s',label=Y_names[1],zorder=10,s=15)
axs[1].scatter(X[Y==2,0],X[Y==2,1],marker='d',label=Y_names[2],zorder=10,s=15)
axs[1].set_title('Softmax $C=$'+str(C))
plt.tight_layout()
plt.savefig('Contours',dpi=700)
    
#%%Calculate metrics w/ SKlearn

classifiers = ['OvR LogReg','Softmax']
ovo_ovr = ['OneVsRest','Softmax']
for i in range(len(classifiers)):
    print(classifiers[i])
    accuracy_SK = sk.metrics.accuracy_score(Y_test,y_pred[i,:])
    precision_SK = sk.metrics.precision_score(Y_test,y_pred[i,:],average=None)
    recall_SK = sk.metrics.recall_score(Y_test,y_pred[i,:],average=None)
    F1 = sk.metrics.f1_score(Y_test,y_pred[i,:],average=None) 
    print('Accuracy: '+ str(accuracy_SK))
    print('Precision: '+str(precision_SK))
    print('Recall: '+str(recall_SK))
    print('F1 Score: '+str(F1)+'\n') 

#%% Building Conf. Matrix

cm = []
for i in range(len(classifiers)):
    cm_new = confusion_matrix(Y_test,y_pred[i,:])
    cm_new = np.insert(cm_new,0,[0,0,0],axis=0)
    cm_new = np.insert(cm_new,0,[0,0,0,0],axis=1)
    cm.append(cm_new)
    
fig1, axs = plt.subplots(1,2, sharey = True, sharex = True, figsize=(6.5,4))
for (j,i),label in np.ndenumerate(cm[0]):
    if i != 0 and j != 0:
        if label <= 7:
            axs[0].text(i,j,label,ha='center',va='center',color='white')
        else:
            axs[0].text(i,j,label,ha='center',va='center')
        
for (j,i),label in np.ndenumerate(cm[1]):
    if j != 0 and i !=0:
        if label <= 7:
            axs[1].text(i,j,label,ha='center',va='center',color='white')
        else:
            axs[1].text(i,j,label,ha='center',va='center')       
axs[0].imshow(cm[0],cmap='gray')
axs[0].set_title('OvR Log Reg')
axs[0].set_xlim(0.5,3.5)
axs[0].set_ylim(0.5,3.5)
axs[0].set_xticks([1,2,3])
axs[0].set_yticks([1,2,3])
axs[1].imshow(cm[1],cmap='gray')
axs[1].set_title('Softmax')
axs[1].set_xticks([1,2,3])
axs[1].set_yticks([1,2,3])
axs[1].set_xlim(0.5,3.5)
axs[1].set_ylim(0.5,3.5)
fig1.suptitle('OvR Vs Softmax Confusion Matrices',fontsize=12)
axs[0].set_ylabel('True Values',fontsize=12)
axs[0].set_xlabel('Predicted Values',fontsize=12)
plt.tight_layout()
plt.savefig('ConfusionMatrices',dpi = 600)