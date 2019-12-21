import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import JokeRsNN
from config import *
from preprocessing import data_preprocess
from text_processing import *


def get_embedding_matrix(frame):
    uid = torch.from_numpy(frame['uid'].values)
    u = get_user_embedding(uid)

    jid = torch.from_numpy(frame['jid'].values)
    j = get_joke_embedding(jid)

    if torch.cuda.is_available():
        u = u.cuda()
        j = j.cuda()
    return u, j


def get_user_embedding(uid):
    uid_embed_layer = nn.Embedding(UID_NUM+1, 32)
    uid_embed_matrix = uid_embed_layer(uid).float()
    return uid_embed_matrix


def get_joke_embedding(jid):
    jid_embed_layer = nn.Embedding(JOKE_NUM+1, 32)
    jid_embed_matrix = jid_embed_layer(jid)
    return jid_embed_matrix


if __name__ == '__main__':
    loss_sq = 0.0
    loss = 0.0

    jokes = read_text()
    ratings = read_ratings()
    jokes_sim = [[0] * 150] * 150
    jokes_sim = tf_idf(jokes, jokes_sim)
    ratings_with_sim = [[0] * 150] * 500
    ratings_with_sim = predict(ratings, jokes_sim, ratings_with_sim)

    count = 0
    for i in range(500):
        for j in range(150):
            if ratings[i][j] != 99:
                loss_sq = (ratings[i][j] - ratings_with_sim[i][j]) ** 2
                loss += abs(ratings[i][j] - ratings_with_sim[i][j])
                count += 1

    rmse_sim = math.sqrt(loss_sq / (150 * 500 - count))
    mae_sim = loss / (150 * 500 - count)
    nmae_sim = mae_sim / 20

    print('the RMSE with sim is %f' % rmse_sim)
    print('the MAE with sim is %f' % mae_sim)
    print('the NMAE with sim is %f' % nmae_sim)

    X_train, y_train, X_test, y_test = data_preprocess('/Users/abe/Desktop/joke_recomm_sys_mlp/dataset'
                                                       '/jester_dataset_2')
    use_gpu = torch.cuda.is_available()
    net = JokeRsNN()
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # Optimizer
    loss_func = nn.MSELoss()  # Loss Function

    for epoch in range(64):
        running_loss = 0.0
        i = 0
        while i < X_train.shape[0]:
            optimizer.zero_grad()
            batch_end = i + BATCH_SIZE
            if batch_end >= X_train.shape[0]:
                batch_end = X_train.shape[0]
            u, j = get_embedding_matrix(X_train[i: batch_end])
            y = y_train[i: batch_end]
            y = torch.from_numpy(y.values).float()
            if use_gpu:
                y = y.cuda()
            out = net.forward(u, j)
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()

            # print('[%d], loss is %f' % (epoch, loss.data[0]))
            running_loss += loss.item()
            i = batch_end
        temp = X_train.shape[0] // BATCH_SIZE
        avg_running_loss = running_loss / temp
        print('epoch [%d] finished, the average loss is %f' % (epoch, avg_running_loss))

    loss_sum_sq = 0.0
    loss_sum = 0.0
    new_loss_sum = 0.0

    i = 0
    u, j = get_embedding_matrix(X_test)
    y = torch.from_numpy(y_test.values).float()
    if use_gpu:
        y = y.cuda()

    # w = 0
    # lap = 1
    # while w <= 1:
    #     c = 0
    #     for i in range(X_test.shape[0]):
    #         out = net.forward(u[i].unsqueeze(0), j[i].unsqueeze(0))
    #         loss_sum_sq += (out - y[i]) ** 2
    #         loss_sum += abs(out - y[i])
    #
    #         if X_train[i][0] <= 500:
    #             new_out = w * out + (1 - w) * ratings_with_sim[X_train[i, 0]-1][X_train[i, 1]-1]
    #             new_loss_sum += abs(new_out - float(y[i]))
    #             c += 1
    #
    #     new_mae = new_loss_sum / c
    #     new_nmae = new_mae / 20
    #
    #     print('No. %d new MAE is %f' % (lap, new_mae))
    #     print('No. %d new NMAE is %f' % (lap, new_nmae))
    #
    #     w += 0.05
    #     lap += 1

    for i in range(X_test.shape[0]):
        out = net.forward(u[i].unsqueeze(0), j[i].unsqueeze(0))
        loss_sum += (out - y[i]) ** 2

    rmse = math.sqrt(loss_sum_sq / X_test.shape[0])
    mae = loss_sum / X_test.shape[0]
    nmae = mae / 20

    print('the RMSE is %f' % rmse)
    print('the MAE is %f' % mae)
    print('the NMAE is %f' % nmae)

