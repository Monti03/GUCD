from calendar import EPOCH
from tkinter import E
import tensorflow as tf

from constants import LR, l, gamma, eta, beta, DATASET_NAME, epochs
from utils import load_data
from trainer import Trainer

import getopt, sys

# Get full command-line arguments
full_cmd_arguments = sys.argv

argument_list = full_cmd_arguments[1:]

print(argument_list)

short_options = "d:l:g:e:b:"
long_options = ["dataset=", "lr=", "lambda=", "gamma=", "eta=", "beta=", "epochs="]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    # Output error, and return with an error code
    print (str(err))
    sys.exit(2)

# Evaluate given options
for current_argument, current_value in arguments:
    if current_argument in ("-d", "--dataset"):
        DATASET_NAME = current_value
    elif current_argument in ("--lr"):
        LR = float(current_value)
    elif current_argument in ("l", "--lambda"):
        l = float(current_value)
    elif current_argument in ("g","--gamma"):
        gamma = float(current_value)
    elif current_argument in ("e","--eta"):
        eta = float(current_value)
    elif current_argument in ("b","--beta"):
        beta = float(current_value)
    elif current_argument in ("--epochs"):
        epochs = int(current_value)
    else:
        raise Exception(f"No such parameter named {current_argument}")
    

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
    trainer.initialize_data(features, complete_adj, gamma, eta, beta, l, n_clusters, LR, labels, epochs)
    trainer.train()