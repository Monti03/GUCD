import tensorflow as tf
import math 

# loss wrt the attributes
def attribute_loss(y_actual,y_pred):      
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_actual, y_pred)

# loss wrt the predicted links
def topological_loss(y_actual,y_pred):
    return tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(y_actual, y_pred)

# with this loss we are trying to force the fact that if two
# nodes have similar neighbours and attributes,
# should have a similar embedding
def reg_loss(F, S, Z):
    phi = tf.subtract(F, S)
    tmp = tf.matmul(Z, phi, transpose_a=True)
    res = tf.matmul(tmp, Z)
    
    return 2*tf.linalg.trace(res)   

# returns gamma*top_loss + (1-gamma)*att_loss + eta*r_loss
def total_loss(y_actual,y_pred, F, S, Z, gamma, eta):
    top_loss = topological_loss(y_actual[0], y_pred[0])
    att_loss = attribute_loss(y_actual[1], y_pred[1])
    r_loss = reg_loss(F, S, Z)

    print("top_loss:{}, att_loss:{}, reg_loss:{}".format(gamma*top_loss, (1-gamma)*att_loss, eta*r_loss))

    return gamma*top_loss + (1-gamma)*att_loss + eta*r_loss
