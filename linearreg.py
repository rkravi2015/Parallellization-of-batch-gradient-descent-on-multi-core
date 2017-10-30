import pandas as pd
import numpy as np
import time
def gradient_descent(X, y, theta, alpha, iterations):
    for iteration in range(iterations):
        gradient = (X.transpose())*((X*theta)-y)
        theta = theta - alpha*gradient/X.shape[0]
        print("Iteration: ",iteration+1," Line: ",float(theta[0])," + ",float(theta[1]),"x                           ",end="\r")
    print("\n")
    return theta

data = pd.read_csv('ex1data1.txt', names = ['x', 'y'])
X_df = pd.DataFrame(data.x)
y_df = pd.DataFrame(data.y)
iterations = 1500
alpha = 0.01
X = np.c_[np.ones((X_df.shape[0],1)),np.matrix(X_df)]
y = np.matrix(y_df)
theta = np.matrix(np.zeros((2,1)))
time1=time.time()
t = gradient_descent(X,y,theta,alpha, iterations)
time2=time.time()-time1
print("Time taken: ",time2*1000," ms")
print(t)