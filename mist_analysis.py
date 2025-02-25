import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
dataset_path = r"C:\Users\Asus\Desktop\train.csv\train.csv"
data = pd.read_csv(dataset_path)
#print(data.head())


data=np.array(data)
m,n=data.shape
np.random.shuffle(data)

data_dev=data[0:1000].T
Y_dev=data_dev[0]
X_dev=data_dev[1:n]
X_dev=X_dev/255

data_train=data[1000:m].T
Y_train=data_train[0]
X_train=data_train[1:n]
X_train=X_train/255.
_,m_train=X_train.shape

print(Y_dev)
print(Y_train)


def init_parameters():
    w1=np.random.rand(10,784)-0.5
    b1=np.random.rand(10,1)-0.5
    w2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5
    return w1,b1,w2,b2

def relu(z):
    return np.maximum(z,0)

def softmax(z):
    a=np.np.exp(z)/sum(np.exp(z))
    return a

def forward_prop(w1,b1,w2,b2,x):
    z1=w1.dot(x)+b1 
    a1=relu(z1)
    z2=w2.dot(a1)+b2
    a2=softmax(z2)  
    return z1,a1,z2,a2

def relu_deriv(z):
    return z>0

def one_hot(y):
    one_hot_y=np.zeros((y.size, y.max()+1))
    one_hot_y[np.arange(y.size),y]=1
    one_hot_y=one_hot_y.T
    return one_hot_y

def bwd_prop(z1,a1,z2,a2,w1,w2,x,y):
    one_hot_y=one_hot(y)
    dz2=a2-one_hot_y
    dw2=1/m*dz2.dot(a1.T)
    db2=1/m*np.sum(dz2)
    dz1=w2.T.dot(dz2)*relu_deriv(z1)
    dw1=1/m*dz1.dot(x.T)
    db1=1/m*np.sum(dz1)
    return dw1,db1,dw2,db2

def update_para(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1=w1-alpha*dw1
    b1=b1-alpha*db1
    w2=w2-alpha*dw2
    b2=b2-alpha*db2
    return w1,b1,w2,b2

def get_prediction(a2):
    return np.argmax(a2,0)

def get_accuracy(prediction,y):
    print(prediction,y)
    return np.sum(prediction==y)/y.size

def gradient_descent(x,y,alpha,iterations):
    w1,b1,w2,b2=init_parameters()
    for i in range(iterations):
        z1,a1,z2,a2=forward_prop(w1,b1,w2,b2,x)
        dw1,db1,dw2,db2=bwd_prop(z1,a1,z2,a2,w1,w2,x,y)
        w1,b1,w2,b2=update_para(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha)
        if i % 10==0:
            print("Iterations:",i)
            predictions=get_prediction(a2)
            print(get_accuracy(predictions,y))
    return w1,b1,w2,b2








