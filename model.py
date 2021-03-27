import tensorflow as tf
from layers import *
from constants import *

class MyModel(tf.keras.Model):

    def __init__(self, Y, K, D, adj_n, m):
        super(MyModel, self).__init__()
        # define the layers of the model        
        self.conv_1 = GraphSparseConvolution(adj_norm=adj_n, output_size=CONV1_OUT_SIZE, dropout_rate=DROPOUT, act=tf.nn.relu)
        self.conv_2 = GraphConvolution(adj_norm=adj_n, output_size=K, dropout_rate=DROPOUT, act=tf.nn.softmax)
        self.mrf = MRFLayer(Y, K, act=tf.nn.softmax)
        self.top_dec = TopologyDecoder(D, K, act=tf.math.sigmoid)
        self.att_dec = AttributesDecoder(m, K, act=lambda x:x)

    def call(self, inputs):
        # get the result of the first convolution
        x = self.conv_1(inputs)
        # get the result of the second convolution
        self.x2 = self.conv_2(x)
        # get the embedding by using the MRF layer
        self.z = self.mrf(self.x2)
        # decode the embedding and obtain topology and attributes
        att = self.att_dec(self.z)
        top = self.top_dec(self.z)

        # return topology and attributes decoded
        return [tf.reshape(top, [-1]), att]

    def getZ(self):
        return self.z
    
    def getX2(self):
        return self.x2