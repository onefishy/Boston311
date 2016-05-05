import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
sns.set_style("white")

import time
import timeit

import scipy.stats 
from scipy.io import loadmat, savemat
from scipy.stats import gaussian_kde
from scipy.interpolate import spline
import pandas as pd
import pymc as pm
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal as MVN

import re
import numpy as np


########################################
########################################
###   K-means init for mu and Sigma  ###
########################################
########################################

def k_means(N, D, K, X):
    kmeans = KMeans(init='random', n_clusters=K, n_init=K) #k-means++
    kmeans.fit(data)
    mu = kmeans.cluster_centers_
    labels = kmeans.predict(X)
    
    clusters = [labels[labels == k].shape[0] for k in xrange(K)]    
    pi = np.array([clusters[k] / (1. * sum(clusters)) for k in xrange(K)])
    
    clusters = [[X[n] for n in xrange(N) if labels[n] == k] for k in xrange(K)]
    
    Sigma = []
    for k in xrange(K):
        Sigma_k = np.zeros((D, D))
        for pt in clusters[k]:
            Sigma_k += np.outer(pt, pt)
        Sigma.append(Sigma_k - len(clusters[k]) * np.outer(mu[k], mu[k]))
        
    return pi, mu, Sigma

########################################
########################################
#####         EM for MLE           #####
########################################
########################################

def MLE_EM(K, D, N, X, pi_0, mu_0, Sigma_0, iters):
    
    #initialization of parameters    
    r = np.zeros((N, K)) #responsibility of each cluster for each data pt
    pi = pi_0 #mixture coefficients
    mu = mu_0 #means (K x D) matrix
    Sigma = Sigma_0 #coveriances length-k list of (K x K) matrices
    
    #---------------- Likelihood ----------------#    
    def lkhd(pi, mu, Sigma):
        prob = np.zeros((N, K))
        for k in xrange(K):
            prob[:, k] = pi[k] * MVN.pdf(X, mu[k, :], Sigma[k])
        return np.nan_to_num(prob)
    
    #---------------- E-Step ----------------#
    def E_step():
        prob = lkhd(pi, mu, Sigma) 
        return np.nan_to_num(np.diag(np.reciprocal(np.sum(prob, axis=1))).dot(prob))
    
    #---------------- M-Step ----------------#
    def M_step():
        r_ks = np.sum(r, axis=0)
        pi_new = 1. / N * r_ks
        mu_new = np.nan_to_num(np.diag(np.reciprocal(r_ks)).dot(r.T.dot(X)))
        
        Sigma_new = []
        for k in xrange(K):
            Sigma_k = np.zeros((D, D))
            for n in xrange(N):
                Sigma_k += r[n, k] * np.outer(X[n, :] - mu[k, :], X[n, :] - mu[k, :])
            Sigma_new.append(np.nan_to_num(Sigma_k / r_ks[k]))
            
        return pi_new, mu_new, Sigma_new
    
    #---------------- Alternate Between E and M-steps ----------------#
    for i in xrange(iters):        
        r = E_step()
        pi, mu, Sigma = M_step()
        
    r = E_step()
    
    return pi, mu, Sigma, r

########################################
########################################
#####         EM for MAP           #####
########################################
########################################

def MAP_EM(K, D, N, X, pi_0, mu_0, Sigma_0, S_0, m_0, nu_0, beta_0, alpha_0, iters):
    
    #initialization of intermediate parameters
    r = np.zeros((N, K)) #responsibility of each cluster for each data pt
    pi = pi_0 #mixture coefficients
    mu = mu_0 #means (K x D) matrix
    Sigma = Sigma_0 #coveriances length-k list of (K x K) matrices
    S = [np.eye(D) for k in xrange(K)] #(K x K) scatter matrices
    X_mean = mu_0 #mean of pts in each cluster (K x D) matrix
    
    #---------------- Likelihood ----------------#    
    def lkhd(pi, mu, Sigma):
        prob = np.zeros((N, K))
        for k in xrange(K):
            prob[:, k] = pi[k] * MVN.pdf(X, mu[k, :], Sigma[k]) 
        return prob
    
    #---------------- E-Step ----------------#
    def E_step():
        prob = lkhd(pi, mu, Sigma) 
        return np.diag(np.reciprocal(np.sum(prob, axis=1))).dot(prob)
        
    
    #---------------- M-Step ----------------#
    def M_step():
        r_ks = np.sum(r, axis=0)
        pi_new = (r_ks + alpha_0 - 1) * 1. / (N + np.sum(alpha_0) - K)
        X_mean_new = np.nan_to_num(np.diag(np.reciprocal(r_ks)).dot(r.T.dot(X)))
        mu_new = np.nan_to_num(np.diag(np.reciprocal(r_ks 
                                                     + beta_0)).dot(np.diag(r_ks).dot(X_mean) 
                                                                    + beta_0 * m_0))
        S_new = []
        Sigma_new = []
        for k in xrange(K):            
            c_1 = (beta_0 * r_ks[k]) / (beta_0 + r_ks[k])
            c_2 = nu_0 + r_ks[k] + D + 2
            Sigma_k = np.nan_to_num(S_0 + S[k] + c_1 
                                    * np.outer(X_mean[k, :] - m_0, X_mean[k, :] - m_0))
            Sigma_new.append(Sigma_k * 1./c_2)
            
            S_k = np.zeros((D, D))
            for n in xrange(N):
                S_k += r[n, k] * np.outer(X[n, :] - X_mean[k, :], X[n, :] - X_mean[k, :])
                
            S_new.append(S_k)
        return pi_new, X_mean_new, mu_new, S_new, Sigma_new
        
    #---------------- Alternate Between E and M-steps ----------------#
    for i in xrange(iters): 
        r = np.nan_to_num(E_step())
        pi, X_mean, mu, S, Sigma = M_step()
        
    r = E_step()
    
    return pi, X_mean, mu, S, Sigma, r

########################################
########################################
#####       Plot clusters          #####
########################################
########################################
def plot_clusters(data_std, r, K, title):
    
    for n in xrange(N):
        r_n = r[n, :]
        r_n[r_n < np.max(r_n)] = 0
        r_n[r_n == np.max(r_n)] = 1
        r[n, :] = r_n
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    clusters = [np.asarray([data_std[n] for n in xrange(N) if r[n, k] == 1]) for k in xrange(K)]

    values = np.linspace(0, 1, K)
    cm = plt.get_cmap('rainbow') 
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    for k in xrange(K):    
        if len(clusters[k]) > 0:
            colorVal = scalarMap.to_rgba(values[k])
            ax.scatter(clusters[k][:, 0], clusters[k][:, 1], clusters[k][:, 2], c=colorVal)
   
    plt.title(title)
    plt.show()

########################################
########################################
#####     Data visualization       #####
########################################
########################################


by_reasons = loadmat('2015_data_by_reason.mat')['data'][:, :182]
x = np.linspace(0, len(by_reasons[0]), len(by_reasons[0]))

adjusted = np.zeros(by_reasons.shape)

for i in xrange(len(by_reasons)):
    for j in xrange(i + 1):
        adjusted[i, :] += by_reasons[j, :]
    adjusted[i, :] += 1
adjusted[0, :] = by_reasons[0, :]

values = np.linspace(0, len(adjusted) + 1, len(adjusted) + 1)
cm = plt.get_cmap('prism') 
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

xnew = np.linspace(x.min(), x.max(), 600)

adjusted_sm = np.zeros((adjusted.shape[0], 600))
for i in xrange(len(adjusted)):
	adjusted_sm[i, :] = spline(x, adjusted[i, :], xnew)

colorVal = scalarMap.to_rgba(values[0])
zero = np.zeros(adjusted_sm.shape[1])
plt.fill_between(xnew, np.sqrt(adjusted_sm[0,:]), zero, color=colorVal)
frame1 = plt.gca()
for i in xrange(1, len(adjusted)):
    colorVal = scalarMap.to_rgba(values[i])
    plt.fill_between(xnew, np.sqrt(adjusted_sm[i,:]), np.sqrt(adjusted_sm[i - 1,:]), color=colorVal)

plt.xlim(0, 182)
plt.ylim(0, 47)
frame1.axes.get_yaxis().set_visible(False)
xlabels = [''] * (600)
xlabels[50] = 'January'
xlabels[200 - 50] = 'February'
xlabels[300 - 50] = 'March'
xlabels[400 - 50] = 'April'
xlabels[500 - 50] = 'May'
xlabels[600 - 50] = 'June'
plt.xticks(xnew, xlabels, rotation=30)
sns.despine(left=True, bottom=True, right=True)
plt.title('311 Requests January thru June, 2015')


data = loadmat('2015_data_by_open.mat')['data']
x = data[:, 1]
y = data[:, 2]
xy = np.vstack((x, y))
z = gaussian_kde(xy)(xy)

idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

values = np.linspace(0, 1, len(idx))
cm = plt.get_cmap('rainbow') 
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
colorVal = scalarMap.to_rgba(values)

fig, ax = plt.subplots()
ax.scatter(x, y, c=colorVal, s=50, edgecolor='')
plt.xlim(min(x) - 1e-3, max(x) + 1e-3)
plt.ylim(min(y) - 1e-3, max(y) + 1e-3)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Intensity Surface of 311 Requests in 2015')
plt.show()

