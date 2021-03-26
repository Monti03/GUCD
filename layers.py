import tensorflow as tf

# noise_shape is [num_nonzero_elements], namely it is a list containing the number of elements of the sparse tensor
# keep_prob is 1-dropout_rate 
# inputs is the sparse tensor which we are applying dropout to
def dropout_sparse(inputs, keep_prob, noise_shape):
    keep_tensor = keep_prob + tf.random.uniform(noise_shape)
    to_retain = tf.cast(tf.floor(keep_tensor), dtype=tf.bool)
    out = tf.sparse.retain(inputs, to_retain=to_retain)

    # the elements of the tensor are rescaled after dropout
    return out * (1/keep_prob)

class GraphSparseConvolution(tf.keras.layers.Layer):
    # the graph takes as input the adjacency matrix in the form of a sparse tensor
    # it is important that the type of the elements of the tensor are explicitly converted to floats
    def __init__(self, adj_norm, output_size=32, dropout_rate=0.0, act=tf.nn.relu):
        super(GraphSparseConvolution, self).__init__()
        self.adj_norm = adj_norm
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.act = act


    # input_shape here will be automatically set as the shape of the input tensor, that will be the feature matrix
    def build(self, input_shape):
        init = tf.keras.initializers.GlorotNormal(seed=None)
        self.kernel = self.add_weight('kernel', initializer=init, shape=[int(input_shape[-1]),self.output_size])

    # the input is a sparse tensor whose elements have been explicitly converted to floats
    def call(self, inputs):
        x = inputs
        x = dropout_sparse(inputs, 1-self.dropout_rate, [len(inputs.values)])
        x = tf.sparse.sparse_dense_matmul(x, self.kernel)
        x = tf.sparse.sparse_dense_matmul(self.adj_norm, x)
        outputs = self.act(x)
        return outputs

# the only difference between these two classes is that the first will treat also the features as a sparse matrix (the adjacency matrix will always be sparse)
class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, adj_norm, output_size=16, dropout_rate=0.0, act=tf.nn.softmax):
        super(GraphConvolution, self).__init__()
        self.adj_norm = adj_norm
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.act= act
    
    def build(self, input_shape):
        init = tf.keras.initializers.GlorotNormal(seed=None)
        self.kernel = self.add_weight('kernel', initializer=init, shape=[int(input_shape[-1]),self.output_size])


    # the input to the call function is a dense tensor whose elements have been explicitly converted to floats
    def call(self, inputs):
        x = inputs
        x = tf.keras.layers.Dropout(rate=self.dropout_rate)(inputs)
        x = tf.matmul(x, self.kernel)
        x = tf.sparse.sparse_dense_matmul(self.adj_norm, x)
        outputs = self.act(x)
        return outputs

class MRFLayer(tf.keras.layers.Layer):
    # beta is the tradeoff parameter that balances topology and attributes
    # Y \in R^{n,n}: 
    #   Y_{i,j} = beta \frac{d_i d_j}{2e} + (1-beta)\frac{cos_sim(att(v_i), att(v_j))}{\sum_k cos_sim(att(v_i), att(v_k))} 
    
    # K is the number of clusters
    def __init__(self, Y, K, dropout_rate=0, act=tf.nn.softmax):
        super(MRFLayer, self).__init__()
        self.Y = Y
        self.K = K
        self.dropout_rate = dropout_rate
        self.act = act
    
    def build(self, input_shape):
        init = tf.keras.initializers.GlorotNormal(seed=None)
        self.MRFLayer_weights = self.add_weight('MRFLayer_weights', initializer=init, shape=[self.K, self.K])
    
    # returns act(inputs - Y*inputs*MRFLayer_weights)
    def call(self, inputs):
        inputs = tf.keras.layers.Dropout(rate=self.dropout_rate)(inputs)
        x = inputs
        x = tf.matmul(x, self.MRFLayer_weights)
        x = tf.matmul(self.Y, x)
        x = tf.subtract(inputs, x)  
        return self.act(x)

class TopologyDecoder(tf.keras.layers.Layer):
    
    # K number of clusters
    # D diagonal matrix with values deg1,..., degn
    def __init__(self, D, K, act=tf.math.sigmoid):
        super(TopologyDecoder, self).__init__()
        self.K = K
        self.D = D
        self.act = act
        
    def build(self, input_shape):
        init = tf.keras.initializers.GlorotNormal(seed=None)
        self.TopologyDecoder_weights = self.add_weight('TopologyDecoder_weights', initializer=init, shape=[self.K, self.K])
    
    # returns (D*Z*TopologyDecoder_weights*Z^T*D^T)
    # the transpose of D makes no sense since it is diagonal, but the paper does this
    def call(self, inputs):
        x = inputs
        x = tf.matmul(self.D, x)
        x = tf.matmul(x, self.TopologyDecoder_weights)
        x = tf.matmul(x, inputs, transpose_b = True)
        x = tf.matmul(x, self.D, transpose_b = True)
        return self.act(x)

# constraint used in the attribute decoder
# sum of the cols of the weight matrix = 1
class SumToOne(tf.keras.constraints.Constraint):

  def __call__(self, w):
    col_sums = tf.reduce_sum(w, axis=0)
    return tf.transpose(tf.divide(tf.transpose(w), tf.reshape(col_sums, (-1, 1))))

  def get_config(self):
    return {}

class AttributesDecoder(tf.keras.layers.Layer):

    # K number of clusters
    # m is the number of features
    def __init__(self, m, K, act=tf.nn.softmax):
        super(AttributesDecoder, self).__init__()
        self.m = m
        self.K = K
    
    def build(self, input_shape):
        init = tf.keras.initializers.GlorotNormal(seed=None)
        self.AttributesDecoder_weights = self.add_weight('AttributesDecoder_weights', constraint=SumToOne(), initializer=init, shape=[self.K, self.m])
    
    # returns (Z*AttributesDecoder_weights)
    def call(self, inputs):
        return tf.matmul(inputs, self.AttributesDecoder_weights)
        