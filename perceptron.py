import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)
num_observations=500

x1=np.random.multivariate_normal([2,-2],[[1,.75],[.75,1]],num_observations)
x2=np.random.multivariate_normal([-2,2],[[1,.75],[.75,1]],num_observations)
X=np.vstack((x1,x2)).astype(np.float32) # (1000,2)
Y=np.hstack((np.zeros(num_observations),np.ones(num_observations)))
Y=np.reshape(Y,(np.shape(Y)[0],1)) # (1000,1)

W=np.zeros((2,1))
threshold=5e-3
alpha=0.2

prediction=np.zeros((Y.shape[0],Y.shape[1]))
for i in range(1000):
    for j in range(Y.shape[0]):
        temp=np.dot(W.T,X[j])
        prediction[j]=temp>0
        for k in range(2):
            W[k]+=alpha*(Y[j]-prediction[j])*X[j][k]
    error=np.sum(np.absolute(prediction-Y))/Y.shape[0]
    #print(error)
    if error<threshold:
        break

print('error =',error,'\n','w =',W)
xx1 = np.array([t[0] for t in x1])
yy1 = np.array([t[1] for t in x1])
xx2 = np.array([t[0] for t in x2])
yy2 = np.array([t[1] for t in x2])
plt.scatter(xx1,yy1,s=10,c='r')
plt.scatter(xx2,yy2,s=10,c='b')
#plt.show()
x_linear = np.linspace(-2,2,2*num_observations)
y_linear = (-W[0,0]/W[1,0])*x_linear
plt.plot(x_linear,y_linear)
plt.show()
