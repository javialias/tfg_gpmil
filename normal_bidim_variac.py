#Código para las distribuciones original y aproximada en el caso de inferencia variacional 
# para una gaussiana bidimensional.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
 
 
plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize']=14,5
fig = plt.figure()
 
random_seed=1000
cov_val = [0]
 

#mean = mu.reshape(2,)
meandef=np.array([0,0])
mean = meandef.reshape(2,) 
pdf_list = []
 
for idx, val in enumerate(cov_val):
     
    #cov = Sigma
    #cov = vwpost
    cov=np.matrix([[2,0],[0,1]])
    distr = multivariate_normal(cov = cov, mean = mean,
                                seed = random_seed)
     

    mean_1, mean_2 = mean[0], mean[1]
    sigma_1, sigma_2 = cov[0,0], cov[1,1]
     
    x = np.linspace(-3, 3, num=100)
    y = np.linspace(-3, 3, num=100)
    X, Y = np.meshgrid(x,y)
     

    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])
     
    key = 131+idx
    ax = fig.add_subplot(key, projection = '3d')
    ax.plot_surface(X, Y, pdf, cmap = "Blues")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f'Covariance between x1 and x2 = {val}')
    pdf_list.append(pdf)
    ax.axes.zaxis.set_ticks([])
 
plt.tight_layout()
plt.show()
 
fig=plt.figure()
for idx, val in enumerate(pdf_list):
    plt.subplot(1,3,idx+1)
    plt.contourf(X, Y, val, cmap="Blues")
    #plt.xlabel("x1")
    #plt.ylabel("x2")
    #plt.title(f'')
plt.tight_layout()
plt.show()
fig.savefig("meanfield2", dpi=200)
