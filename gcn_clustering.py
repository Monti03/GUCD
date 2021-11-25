import tensorflow as tf

from constants import LR, l, gamma, eta, beta, DATASET_NAME
from utils import load_data
from trainer import Trainer

if __name__ == "__main__":
    # define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    #load data
    data = load_data(DATASET_NAME)

    complete_adj = data[0]
    features = data[1] 
    labels = [x[1] for x in data[2]]
    n_clusters = data[3]
    
    trainer = Trainer()
    trainer.initialize_data(features, complete_adj, gamma, eta, beta, l, n_clusters, LR, labels, 50)
    trainer.train()