from losses import total_loss
from utils import *
import tensorflow as tf
from metrics import clustering_metrics

from model import MyModel

class Trainer():
    def __init__(self) -> None:
        pass
    # features is the sparse matrix containing the features
    # adj_train is the sparse adjiacency matrix
    # gamma : defines how to balance att loss and topo loss
    # eta defines the imporance of the reg loss
    # beta defines how to balance topo info and att info in MRF layer
    # l lambda defines how to balance Stopo and Satt
    # n_clusters the number of clusters
    # lr is the learning rate
    # clustering_labels -> labels of the nodes
    # epochs
    def initialize_data(self, features, adj_train, gamma, eta, beta, l, n_clusters, lr, clustering_labels, epochs):
        self.number_of_features, self.epochs = features.shape[1], epochs
        self.gamma, self.eta, self.beta = gamma, eta, beta

        self.clustering_labels = clustering_labels

        # normalize the adj
        adj_train_norm = compute_adj_norm(adj_train)
        S = getS(adj_train, features, l)
        F = getF(S)
        Y = getY(adj_train, beta, features)

        # this method needs to have the complete adj_train (not only the triu)
        D = getD(adj_train)

        # covnert matrices to tensors
        self.adj_train_norm_tensor = convert_sparse_matrix_to_sparse_tensor(adj_train_norm)
        self.Y_tensor = tf.convert_to_tensor(Y, dtype="float32")
        self.D_tensor = tf.convert_to_tensor(D, dtype="float32")
        
        self.F_tensor = tf.convert_to_tensor(F, dtype="float32")
        self.S_tensor = tf.convert_to_tensor(S, dtype="float32")
        self.feature_tensor = convert_sparse_matrix_to_sparse_tensor(features)

        # define the ground truth
        self.y_actual = [adj_train.toarray().flatten(), features.toarray()]

        # define the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # define the model
        self.model = MyModel(self.Y_tensor, n_clusters, self.D_tensor, self.adj_train_norm_tensor, self.number_of_features)

    def train_loop(self):
        with tf.GradientTape() as tape:
            pred = self.model(self.feature_tensor)

            # get the embedding of the nodes
            Z = self.model.getZ()
            self.Z_np = self.model.getZ().numpy()

            self.X2_np = self.model.getX2().numpy()

            # calculate the loss
            loss = total_loss(self.y_actual, pred, self.F_tensor, self.S_tensor, Z, self.gamma, self.eta)

            # get the gradients
            grad = tape.gradient(loss, self.model.trainable_variables)
        
        # update the weights of the model by using the precendetly calculated gradients
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        return pred[0], pred[1]    

    def train(self):
        for i in range(self.epochs):
            print(f"epoch:{i}")
            pred_top, pred_att = self.train_loop()

            # measure accuracy on the train edges
            top_acc_function = tf.keras.metrics.BinaryAccuracy()
            top_acc_function.update_state(self.y_actual[0], pred_top)
            top_acc_function = top_acc_function.result().numpy()

            # measure the accuracy on the attributes
            att_acc_function = tf.keras.metrics.BinaryAccuracy()
            att_acc_function.update_state(self.y_actual[1].flatten(), tf.reshape(pred_att, [-1]))
            att_train_accuracy = att_acc_function.result().numpy()
            
            print(f"train top acc: {top_acc_function}")
            print(f"train att acc: {att_train_accuracy}")

            # get the labels from the embedding layer
            pred_labels_z = self.Z_np.argmax(1)
            pred_labels_x = self.X2_np.argmax(1)
            
            # get the accuracy pf the predicted labels
            cm = clustering_metrics(self.clustering_labels, pred_labels_z)
            res = cm.clusteringAcc()
            print("acc_z:{}, f1_z:{}".format(res[0], res[1]))

            cm = clustering_metrics(self.clustering_labels, pred_labels_x)
            res = cm.clusteringAcc()
            print("acc_x:{}, f1_x:{}".format(res[0], res[1]))