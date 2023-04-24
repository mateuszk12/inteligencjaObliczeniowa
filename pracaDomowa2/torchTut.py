# import torch
import numpy as np
#TENSORS BASICS
# x = torch.empty(3,4,5)
# y = torch.rand(2,3)
# z = torch.ones(2,3)
# a = torch.ones(3,4, dtype=torch.double)
#można sobie dodawać, odejmować dzielić i mnożyć znakami normalnie
# print(b,z,y)
#SLICING
# f = torch.rand(3,4)
# print(f,end="\n")
# print(f.view(-1,3).size())
#numpy -> tensor and tensor -> numpy


# a = torch.ones(5)
# # print(a)
# b = a.numpy()
# a.add_(1)
# c = torch.from_numpy(b)
#the tensor and nupy will share the same memory
# print(a,b,c)
# device = torch.device("cuda")
# l = torch.ones(2,2,device=device)
# m = l.to("cpu").numpy()
# print(m,l)
#OBLICZNIE GRADIENTU - PONOĆ WAŻNE
# dokonczyc - https://youtu.be/DbeIqrwb_dE?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# x = torch.randn(10,requires_grad=True)
# print(x)
# y = x + 1
#
# z = y.mean()
# print(y,y.mean())
# z.backward()
# print(x.grad)
# #backpropagation
import torch
import torch.nn as nn
X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)
n_samples, n_features = X.shape
input_size = n_features
output_size = n_features
print(n_samples,n_features)
# w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
# def forward(x):
#     return w*x
# def loss(y,y_pred):
#     return ((y_pred-y)**2).mean()
# print(f'pred: {forward(5):.3f}')
model = nn.Linear
learning_rate = 0.01
n_iters = 70
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w],lr=learning_rate)
for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(Y,y_pred)
    #gradient
    l.backward()
    #update weights
    optimizer.step()
    with torch.no_grad():
        w -= learning_rate*w.grad
    optimizer.zero_grad()
    if epoch % 2 == 0:
        print(epoch+1,w,l)
print(forward(X),X,Y)