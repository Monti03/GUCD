# GUCD
Implementation in TF2 of Community-Centric Graph Convolutional Network for Unsupervised Community Detection

You can find the paper at the following link [https://www.ijcai.org/Proceedings/2020/0486.pdf](https://www.ijcai.org/Proceedings/2020/0486.pdf)

# Instructions
To train the model you have to run the following command:

```
python gcn_clustering.py
```
by using this command you are using the constants defined in the `constants.py` file that are not finetuned. I suggest to try to reduce LR and number of parameters.

**Other Parameters**
- `--dataset=` to define the dataset (one value between `cora` and `citeseer`)
- `--lr=` to define the learning rate
- `--lambda=` to define the lambda constant defined in the paper (the variable that defines how to balance Stopo and Satt)
- `--gamma=` to define the gamma constant defined in the paper (the variable that defines how to balance att loss and topo loss)
- `--eta=` to define the eta constant defined in the paper (the variable that defines the imporance of the reg loss)
- `--beta=` to define the beta constant defined in the paper (the variable that defines how to balance topo info and att info in MRF layer)
- `--epochs=` to define the number of epochs

So you can try to run the following command:
```
python gcn_clustering.py --dataset="cora" --lr=0.0001
```

## Dependencies
```
pip3 install scipy numpy tensorflow pandas networkx seaborn sklearn matplotlib munkres

```