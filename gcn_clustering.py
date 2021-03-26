import os  
import math
import random

from scipy.sparse import csr_matrix
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import numpy as np

import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

from sklearn import metrics

from constants import LR, l, gamma, eta, beta, DATASET_NAME, DROPOUT, epochs, CONV1_OUT_SIZE
from utils import load_data
from metrics import clustering_metrics
from model import MyModel
from losses import total_loss

def convert_sparse_matrix_to_sparse_tensor(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    # get the indices of the non zero values in the matrix
    indices_matrix = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    # get the non zero values of the matrix
    values = sparse_mx.data
    #get the shape
    shape = sparse_mx.shape
    # create a sparse tensor
    sparse_tensor = tf.SparseTensor(indices=indices_matrix, values=values, dense_shape=shape)
    return tf.cast(sparse_tensor, dtype=tf.float32)

# features is the sparse matrix containing the features
# adj_train is the sparse adjiacency matrix
# adj_train_norm = D^{-1/2}(A+I)D^{-1/2}
# train_edges, train_false_edges are edges that are not present in the adj_matrix
# clustering_labels -> labels of the nodes
# optimizer: the optimizer to apply
# F is the diagonal matrix s.t. F_ii = sum_j S_ij
# S = lambda*normalized(Stopo) + (1-lambda)normalized(S att)
# gamma : defines how to balance att loss and topo loss
# eta defines the imporance of the reg loss
# Y: square matrix st y_ij = beta* (di dj/ 2e) + (1-beta)*(cos_sim(feat_i, feat_j)/ sum_k cos_sim(feat_i, feat_k))
# D is the diagonal matrix s.t. d_ii = deg(i) in (A+I)
# K is the number of clusters
def train(features, adj_train, adj_train_norm, train_edges, train_false_edges, clustering_labels , optimizer, F, S, gamma, eta, Y, D, number_of_features,K):
    print("training")
    
    max_acc_x = 0
    max_acc_z = 0

    # covnert matrices to tensors
    adj_train_norm_tensor = convert_sparse_matrix_to_sparse_tensor(adj_train_norm)
    Y_tensor = tf.convert_to_tensor(Y, dtype="float32")
    D_tensor = tf.convert_to_tensor(D, dtype="float32")
    
    F_tensor = tf.convert_to_tensor(F, dtype="float32")
    S_tensor = tf.convert_to_tensor(S, dtype="float32")
    feature_tensor = convert_sparse_matrix_to_sparse_tensor(features)

    # define the model
    model = MyModel(Y_tensor, K, D_tensor, adj_train_norm_tensor, number_of_features)
    
    for i in range(epochs):

        with tf.GradientTape() as tape:
            pred = model(feature_tensor)

            # if you want ot train over the non known edges use this 
            """ train_edges_p_pred = [pred[0][x[0]*adj_train.shape[0]+x[1]] for x in train_edges]
            train_edges_n_pred = [pred[0][x[0]*adj_train.shape[0] +x[1]] for x in train_false_edges]

            train_edges_p_l = [1]*len(train_edges_p_pred)
            train_edges_n_l = [0]*len(train_edges_n_pred)

            pred[0] = train_edges_p_pred + train_edges_n_pred

            y_actual = [train_edges_p_l+train_edges_n_l, features.toarray().flatten()] 
            """
            # define the ground truth
            y_actual = [adj_train.toarray().flatten(), features.toarray().flatten()]

            # get the embedding of the nodes
            Z = model.getZ()
            Z_np = model.getZ().numpy()

            X2 = model.getX2()
            X2_np = model.getX2().numpy()

            # calculate the loss
            loss = total_loss(y_actual, pred, F_tensor, S_tensor, Z, gamma, eta)

            # get the gradients
            grad = tape.gradient(loss, model.trainable_variables)

        # update the weights of the model by using the precendetly calculated gradients
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        print("#"*30)
        print("epoch:{}, train loss: {}".format(i, loss))

        # measure accuracy on the train edges
        top_acc_function = tf.keras.metrics.BinaryAccuracy()
        top_acc_function.update_state(y_actual[0], tf.nn.sigmoid(pred[0]))
        top_train_accuracy = top_acc_function.result().numpy()

        # measure the accuracy on the attributes
        att_acc_function = tf.keras.metrics.BinaryAccuracy()
        att_acc_function.update_state(y_actual[1], pred[1])
        att_train_accuracy = att_acc_function.result().numpy()
        
        print("train top acc: {}".format(top_train_accuracy))
        print("train att acc: {}".format(att_train_accuracy))

        # get the labels from the embedding layer
        pred_labels_z = Z_np.argmax(1)
        pred_labels_x = X2_np.argmax(1)
        
        # get the accuracy pf the predicted labels
        cm = clustering_metrics(labels, pred_labels_z)
        res = cm.clusteringAcc()
        print("acc_z:{}, f1_z:{}".format(res[0], res[1]))
        if(res[0]>max_acc_z):
            max_acc_z = res[0]

        cm = clustering_metrics(labels, pred_labels_x)
        res = cm.clusteringAcc()
        print("acc_x:{}, f1_x:{}".format(res[0], res[1]))

        if(res[0]>max_acc_x):
            max_acc_x = res[0]
    print("max_acc_z:{}".format(max_acc_z))
    print("max_acc_x:{}".format(max_acc_x))

# compute D^{-1/2}(A+I)D^{-1/2}
def compute_adj_norm(adj):
    
    adj_I = adj + sp.eye(adj.shape[0])

    D = np.sum(adj_I, axis=1)
    D_power = sp.diags(np.asarray(np.power(D, -0.5)).reshape(-1))

    adj_norm = D_power.dot(adj_I).dot(D_power)

    return adj_norm

# sparse matrix to tuple (coords of non zero values, values, shape)
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# choose uar false edges from the matrix
def get_false_edges(adj, n):
    print("getting false edges")
    false_edges = []

    while len(false_edges) < n:
        r1 = random.randint(0, adj.shape[0]-1)
        r2 = random.randint(0, adj.shape[0]-1)

        if(adj[r1, r2] == 0 and r1<r2):
            false_edges.append([r1,r2])
            
    return false_edges

# get
# adj_train: triu form of the matrix with removed the train and test edges
# edges -> partition of edges to be used in train and test
def get_test_edges(adj, test_size=0, train_size=0):
    print("getting test edges")
    adj_ = sp.triu(adj)
    coords, _, shape = sparse_to_tuple(adj_)
    
    all_edges = coords
    
    # get the number of train and test edges
    num_train = int(train_size*all_edges.shape[0])
    num_test = int(test_size*all_edges.shape[0])

    # shuffle the edges
    np.random.shuffle(all_edges)
    # get the test edges(positive)
    test_edges = all_edges[:num_test]
    # get the train edges(positive)    
    train_edges = all_edges[num_test:num_test+num_train]
    # get the remaning edges(positive)
    res_edges = all_edges[num_test+num_train:]
    
    # generate false edges for train and test
    false_edges = get_false_edges(adj, test_edges.shape[0]+train_edges.shape[0])
    test_false_edges = false_edges[:test_edges.shape[0]]
    train_false_edges = false_edges[test_edges.shape[0]:]
    
    print("got false edges")
    # create the sparse matrix from the remaning edges
    adj_train = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj.shape)
    
    return adj_train, train_edges, np.array(train_false_edges), test_edges, np.array(test_false_edges) 

def cosine_similarity_graph(a,b):
    # the values are or 0 or 1
    den = math.sqrt(len(a)) * math.sqrt(len(b))
    num = set(a.keys()).intersection(set(b.keys()))
    return num / den

def cosine_similarity(a,b):
    # the values are or 0 or 1
    den = math.sqrt(a.sum()) * math.sqrt(b.sum())
    num = np.dot(a, b)
    if(den == 0):
        return 0
    return num / den

# get the Stopo needed to calculate S
# Stopo: s_ij = cos_sim(neigh_i, neigh_j)
def get_S_topo(adj):
    
    if(os.path.exists('{}_stopo.csv'.format(DATASET_NAME))):
        return np.loadtxt('{}_stopo.csv'.format(DATASET_NAME), delimiter=',')

    print("getting S topo")
    s_topo = np.zeros(adj.shape)

    for i in range(adj.shape[0]):
        for j in range(i, adj.shape[0]):
            cs = cosine_similarity(adj.getrow(i).toarray()[0], adj.getrow(j).toarray()[0])
            s_topo[i][j] = cs
            s_topo[j][i] = cs
    
    np.savetxt('{}_stopo.csv'.format(DATASET_NAME), s_topo, delimiter=',')

    return s_topo

# get the Satt needed to calculate S
# Satt: s_ij = cos_sim(att_i, att_j)
def get_S_att(features_sparse):

    if(os.path.exists('{}_satt.csv'.format(DATASET_NAME))):
        return np.loadtxt('{}_satt.csv'.format(DATASET_NAME), delimiter=',')

    print("getting S att")
    s_att = np.zeros((features_sparse.shape[0], features_sparse.shape[0]))

    for i in range(features_sparse.shape[0]):
        for j in range(i, features_sparse.shape[0]):
            cs = cosine_similarity(features_sparse.getrow(i).toarray()[0], features_sparse.getrow(j).toarray()[0])
            s_att[i][j] = cs
            s_att[j][i] = cs
    
    np.savetxt('{}_satt.csv'.format(DATASET_NAME), s_att, delimiter=',')
    return s_att


def min_max_normalizer(m):
    print("normalizing")
    max_ = m.max()
    min_ = m.min()

    m = m-min_
    return m/(max_-min_)


# lambda*normalized(Stopo) + (1-lambda)normalized(S att)
def getS(adj_train, features, l):

    if(os.path.exists('{}_s.csv'.format(DATASET_NAME))):
        return np.loadtxt('{}_s.csv'.format(DATASET_NAME), delimiter=',')

    print("getting S")
    s_topo = min_max_normalizer(get_S_topo(adj_train))
    print("got S topo")
    s_att = min_max_normalizer(get_S_att(features))
    print("got S att")
    
    S = l*s_topo + (1-l)*s_att

    np.savetxt('{}_s.csv'.format(DATASET_NAME), S, delimiter=',')

    return S

# the diagonal matrix st f_ii = sum_j S_ij
def getF(S):
    values = []
    for row in S:
        values.append(row.sum())

    return np.diag(values)

# degree matrix
def getD(adj_train):
    values = []
    for row in adj_train.toarray():
        # +1 since D is the degree matrix of (A+I)
        values.append(row.sum()+1)

    return np.diag(values)

# Y: square matrix st y_ij = beta* (di dj/ 2e) + (1-beta)*(cos_sim(feat_i, feat_j)/ sum_k cos_sim(feat_i, feat_k))
def getY(adj_train, beta, features_sparse):
    if(os.path.exists('{}_Y.csv'.format(DATASET_NAME))):
        return np.loadtxt('{}_Y.csv'.format(DATASET_NAME), delimiter=',')

    G = nx.Graph(sp.triu(adj_train))
    e = G.number_of_edges()

    S_att = get_S_att(features_sparse)
    Y = np.zeros((adj_train.shape[0], adj_train.shape[0]))
    for i in range(adj_train.shape[0]):
        for j in range(adj_train.shape[0]):
            di = len(G[i])
            dj = len(G[j])
            aij = 0
            if(G.has_edge(i,j)):
                aij = 1
            eps_ij = di*dj/(2*e) - aij
            
            z_ij = S_att[i][j]
            sum_z_i = S_att[i].sum()

            Y[i][j] = beta*eps_ij + (1-beta)*z_ij/sum_z_i

    np.savetxt('{}_Y.csv'.format(DATASET_NAME), Y, delimiter=',')
    return Y
    
if __name__ == "__main__":
    # define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    #load data
    data = load_data(DATASET_NAME)

    complete_adj = data[0]
    features = data[1] 
    labels = [x[1] for x in data[2]]
    n_clusters = data[3]
    # get train data
    adj_train_triu, train_edges, train_false_edges, test_edges, test_false_edges = get_test_edges(complete_adj)
    print("got adj_train")
    # the adj returned by get_test_edges is only triu, we need the complete one
    adj_train = adj_train_triu + adj_train_triu.T

    # normalize the adj
    adj_train_norm = compute_adj_norm(adj_train)
    print("normalized the adj matrix")
    S = getS(adj_train, features, l)
    print("got S")
    F = getF(S)
    print("got F")
    Y = getY(adj_train, beta, features)
    print("got Y")

    # this method needs to have the complete adj_train (not only the triu)
    D = getD(adj_train)
    print("got D")
    
    number_of_features = features.shape[1]

    train(features, adj_train, adj_train_norm, train_edges, train_false_edges, labels, optimizer, F, S, gamma, eta, Y, D, number_of_features, K=n_clusters)
