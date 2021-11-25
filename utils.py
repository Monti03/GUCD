# file from https://github.com/tkipf/gcn/blob/master/gcn/utils.py

import sqlite3
import random
import os  
import math
import random

from scipy.sparse import csr_matrix
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import numpy as np

import tensorflow as tf

from constants import *


def load_data(dataset_name):
    if(dataset_name == "cora"):
        return load_cora_data()
    elif (dataset_name == "citeseer"):
        return load_citeseer_data()
    elif(dataset_name == "cora2"):
        return load_cora2_data()

def load_cora2_data():
    adj_complete = np.loadtxt(open("CORA/W.csv", "rb"), delimiter=",")
    adj_sparse = sp.csr_matrix(adj_complete)

    features = np.loadtxt(open("CORA/fea.csv", "rb"), delimiter=",")
    features_sparse = sp.csr_matrix(features)

    gnd = np.loadtxt(open("CORA/gnd.csv", "rb"), delimiter=",")

    labels = []
    clusters = set()
    for i, cluster in enumerate(gnd):
        labels.append([i, cluster])
        clusters.add(cluster)

    return adj_sparse, features_sparse, labels, len(clusters)

def load_citeseer_data():
    content_d = {}
    with open("citeseer/citeseer.content", "r") as fin_c:
        classes = set()
        docs = set()
        for line in fin_c:
            splitted = line.strip().split('\t')
            if(len(splitted)>2):
                content_d[splitted[0]] = {"features":splitted[1:-1], "class":splitted[-1]}
                classes.add(splitted[-1])
                docs.add(splitted[0])
    
    class_to_id = {}
    for i, class_ in enumerate(list(classes)):
        class_to_id[class_] = i
    
    for i, doc in enumerate(list(docs)):
        content_d[doc]["id"] = i
    
    cites = []
    with open("citeseer/citeseer.cites", "r") as fin_c:
        for line in fin_c:
            splitted = line.strip().split("\t")
            if(len(splitted)>1 and splitted[0] in content_d and splitted[1] in content_d):
                cites.append([content_d[splitted[0]]["id"], content_d[splitted[1]]["id"]])
                cites.append([content_d[splitted[1]]["id"], content_d[splitted[0]]["id"]])
    
    cited = np.array(cites).T
    adj_complete = sp.csr_matrix((np.ones(cited.shape[1]), (cited[0], cited[1])))

    labels = []
    for key in content_d.keys():
        labels.append([content_d[key]["id"], class_to_id[content_d[key]["class"]]]) 

    labels = sorted(labels, key= lambda y: y[0])

    features_ = []
    for key in content_d.keys():
        ws = content_d[key]["features"]
        for i, w in enumerate(ws):
            if(w == "1"):
                features_.append([content_d[key]["id"], i])
    
    features_ = np.array(features_).T

    features = sp.csr_matrix((np.ones(features_.shape[1]), (features_[0], features_[1])))

    return adj_complete, features, labels, len(class_to_id.keys())

def load_cora_data():
    # get cursor from db 
    conn = sqlite3.connect('CORA/cora.db')
    cur = conn.cursor()

    # load data from db
    adj_sparse, features_sparse, labels, n_clusters = load_cora_adj_features(cur)
    
    return adj_sparse, features_sparse, labels, n_clusters

def load_cora_adj_features(cur):
    
    cur.execute("SELECT * FROM paper")
    papers = cur.fetchall()
    cur.execute("SELECT * FROM cites")
    cited = cur.fetchall()
    cur.execute("SELECT * FROM content")
    features = cur.fetchall()

    paper_id_to_int = {}
 
    set_labels = set()

    # associate paper_id to integer
    for i, paper in enumerate(papers):
        paper_id_to_int[paper[0]] = i
        set_labels.add(paper[1])
    
    topic_to_label_index = {}
    for i, label in enumerate(list(set_labels)):
        topic_to_label_index[label] = i
    
    labels = [ [paper_id_to_int[x[0]], topic_to_label_index[x[1]]] for x in papers ]

    cited = np.array([[paper_id_to_int[x[0]],paper_id_to_int[x[1]]] for x in cited]).T

    # get sparse matrix 
    adj_s = sp.csr_matrix((np.ones(cited.shape[1]*2), (np.append(cited[0], cited[1]), np.append(cited[1], cited[0])))).tocsr()

    # get set of words
    words = set()
    for feature in features:
        words.add(feature[1])

    # associate word_id to integer
    word_to_int = {}
    for i, word in enumerate(list(words)):
        word_to_int[word] = i

    features = np.array([ [ paper_id_to_int[x[0]], word_to_int[x[1]] ] for x in features ]).T

    # get sparse matrix
    features_s = sp.csr_matrix((np.ones(features.shape[1]), (features[0], features[1])))

    return adj_s, features_s, labels, len(set_labels)

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