"""
    Author: Azaan Rehman (arehman@andrew.cmu.edu)

    Basic Implementation of AdaBoost for 10-701 F20

"""

import numpy as np
import argparse

class Data():
    def __init__(self, data_train_root, data_test_root):

        self.train_root = data_train_root
        self.test_root = data_test_root

    def load_data(self):

        train_raw = np.loadtxt(open(self.train_root, "rb"), delimiter=",")
        self.train_f = train_raw[:,:-1]
        self.train_l = train_raw[:,-1]

        test_raw = np.loadtxt(open(self.test_root, "rb"), delimiter=",")
        self.test_f = test_raw[:,:-1]
        self.test_l = test_raw[:,-1]

class Ada_boost():
    def __init__(self, num_samples, num_epochs):

        self.weights = np.ones((num_samples,1))

        self.beta = np.zeros((num_epochs))


def arg_parser():

    parser = argparse.ArgumentParser("Argument parser for AdaBoost")
    parser.add_argument("--train_root", type=str, default='../datasets/train_adaboost.csv', help="path to the training data")
    parser.add_argument("--test_root", type=str, default='../datasets/test_adaboost.csv', help="path to the test data")

    parser.add_argument("--num_epochs", type=int, default=15, help='number of epochs to train for')
    parser.add_argument("--step_size", type=float, default=0.01, help='step size for SGD')
    parser.add_argument("--hidden_dim", type=int, default=256, help='dimension for the hidden layer')
    parser.add_argument("--seed", type=int, default=0, help='seed for randomization')
    parser.add_argument("--eps", type=float, default=1e-8, help='eps to prevent negative log')

    args = parser.parse_args()

    return args



if __name__ == "__main__":


    args = arg_parser()

    
    print("-----------------------------------------")
    print("Start Loading Data")
    print("-----------------------------------------")


    data = Data(args.train_root, args.test_root)

    print("-----------------------------------------")
    print("Finish Loading Data")
    print("-----------------------------------------")




