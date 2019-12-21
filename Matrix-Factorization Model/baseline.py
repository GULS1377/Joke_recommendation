import os
import torch
import numpy as np
import csv
import gensim

def pre_process(R):
    n_users = len(R)
    n_items = len(R[0]) - 1
    
    # sorted by freq
    R = R[R[:,0].argsort()][:,1:][::-1]
    
    X = np.zeros(shape=(n_users,n_items), dtype=np.float32)
    masks = np.zeros(shape=(n_users,n_items), dtype=np.float32)
    
    for i in range(n_users):
        for j in range(n_items):
            # not rated
            if R[i][j] != 99:
                masks[i][j] = 1
                X[i][j] = R[i][j]
    return X, masks


try:
    X = np.load('data/X.npy')
    masks = np.load('data/masks.npy')
except:
    with open('ratings.csv', newline='\n') as csvfile:
        R = []
        ratings_reader = list(csv.reader(csvfile, delimiter=','))
        for i in range(len(ratings_reader)):
            m = len(ratings_reader[i])
            R.append([])
            for j in range(m):
                R[i].append(float(ratings_reader[i][j]))

    R = np.array(R, dtype=np.float)
    X, masks = pre_process(R)

    np.save('data/X.npy', X)
    np.save('data/masks.npy', masks)


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.user_fac = torch.nn.Embedding(n_users, n_factors)
        self.user_fac.weight.data.uniform_(0, 0.1)
        self.item_fac = torch.nn.Embedding(n_items, n_factors)
        self.item_fac.weight.data.uniform_(0, 0.1)
        self.user_b = torch.nn.Embedding(n_users, 1)
        self.user_b.weight.data.uniform_(0, 0.005)
        self.item_b = torch.nn.Embedding(n_items, 1)
        self.item_b.weight.data.uniform_(0, 0.005)
        
    def forward(self, u, i):
        pred_b = self.user_b(u).squeeze() + self.item_b(i).squeeze()
        pred = (self.user_fac(u)*self.item_fac(i)).sum(dim=1, keepdim=True) + pred_b
        return pred


X = X[:500, :]
masks = masks[:500, :]
model = MatrixFactorization(500, 150, 3)
# print(model)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

rows, cols = masks.nonzero()
p = np.random.permutation(len(rows))
rows, cols = rows[p], cols[p]

for i in range(3000):
    err = 0
    count = 0
    for row, col in zip(*(rows, cols)):
        # Turn data into tensors         
        # print('({0},{1}) - {2}'.format(row,col,X[row, col]))
        rating = torch.FloatTensor([X_gauge_set[row, col]])
        row = torch.LongTensor([row])
        col = torch.LongTensor([col])
        prediction = model(row, col)
        loss = loss_func(prediction, rating)
        err += loss.data.numpy()
        count += 1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if i % 500 == 0:
        print('*')
        for parameters in model.parameters():
            print(parameters)
        print('err={0}, norm_err={1}'.format(err, (err/count)))

 for parameters in model.parameters():
    print(parameters)

  for row, col in zip(*(rows, cols)):
        # Turn data into tensors
        rating = torch.FloatTensor([X_gauge_set[row, col]])
        row_ = torch.LongTensor([row])
        col_ = torch.LongTensor([col])
        prediction = model(row_, col_)
        loss = loss_func(prediction, rating)
        print('______________')
        print('{0} <--> {1}'.format(X_gauge_set[row, col], prediction))
        print(loss.data.numpy())

"""
def initial_model(n_users, n_items, n_features):
    P = np.random.rand(n_users, n_features)
    Q = np.random.rand(n_items, n_features)
    return P, Q

def prediction(P, Q, i, j):
    return np.dot(P[i], Q[j].T)

def sqr_loss(R, P, Q, i, j):
#     print('i={0}, j={1}'.format(i, j))
#     print('R_ij={0}, pred={1}'.format(R[i][j], prediction(P, Q, i, j)))
#     print('*', R[i][j] - prediction(P, Q, i, j))
    return np.square(R[i][j] - prediction(P, Q, i, j))

def loss(R, masks, P, Q, rg=0.01):
    n_features = len(P[0])
    n = len(P)
    m = len(Q)
    loss = 0
    for i in range(n):
        for j in range(m):
            if masks[i][j] == 1:
                loss += sqr_loss(R, P, Q, i, j)
    loss /= (n * m)
    loss += (rg/2) * (np.sum(np.square(P)) + np.sum(np.square(Q)))
    return loss

def gradient(R, P, Q, i, j):
    n_features = len(P[0])
    g_pi = np.zeros(n_features)
    g_qj = np.zeros(n_features)
    cost = R[i][j] - prediction(P, Q, i, j)
    for k in range(n_features):
        g_pi[k] = (-2) * cost * Q[j][k]
        g_qj[k] = (-2) * cost * P[i][k]
    return g_pi, g_qj

def update(R, masks, P, Q, lr=0.0002, rg=0.01):
    n_users = len(R)
    n_items = len(R[0])
    for i in range(n_users):
        for j in range(n_items):
            if masks[i][j] != 0:
                g_pi, g_qj = gradient(R, P, Q, i, j)
                P[i] -= lr * (g_pi + rg * P[i])
                Q[j] -= lr * (g_qj + rg * Q[j])
    return P, Q

def run(X, masks, k, steps=20000, lr=0.0002, rg=0.02):
    n = len(X)
    m = len(X[0])
    P, Q = initial_model(n, m, k)
    for s in range(steps):
        P, Q = update(X, masks, P, Q, lr, rg)
        err = loss(X, masks, P, Q, 0)
        if s % 1000 == 0:
            print('loss =', err)
        if err < 0.001:
            break
    return X, P, Q

with open('ratings.csv', newline='\n') as csvfile:
    R = []
    ratings_reader = list(csv.reader(csvfile, delimiter=','))
    for i in range(len(ratings_reader)):
        m = len(ratings_reader[i])
        R.append([])
        for j in range(m):
            R[i].append(float(ratings_reader[i][j]))

R = np.array(R, dtype=np.float)
X, masks = pre_process(R)
X = X[:20,:]
masks = masks[:20,:]
J7_8 = X[:,6:8]
J13 = X[:,12:13]
J15_19 = X[:,14:19]
m7_8 = masks[:,6:8]
m13 = masks[:,12:13]
m15_19 = masks[:,14:19]
X = np.hstack([J7_8, J13, J15_19])
masks = np.hstack([m7_8, m13, m15_19])
# print(X.shape)
# print(masks)
X, P, Q = run(X, masks, 2)
print(X)
print(np.dot(P, Q.T))

"""
