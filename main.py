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

        self.load_data()

    def load_data(self):

        train_raw = np.loadtxt(open(self.train_root, "rb"), delimiter=",")
        self.train_f = train_raw[:,:-1]
        self.train_l = train_raw[:,-1]

        test_raw = np.loadtxt(open(self.test_root, "rb"), delimiter=",")
        self.test_f = test_raw[:,:-1]
        self.test_l = test_raw[:,-1]

class Ada_boost():
    def __init__(self, train_f_shape, num_epochs):

        self.num_epochs = num_epochs

        self.D = np.ones((train_f_shape[0])) 
        self.D_sum = np.sum(self.D)

        self.beta = np.zeros((num_epochs))
        self.weaks = np.zeros((num_epochs, 3))

    def pred_trans(self, prem_pred):

        return prem_pred*2-1

    def best_weak(self, data):

        train_f = np.copy(data.train_f)

        best_e = np.inf

        best_dir = 0
        best_dim = 0
        best_val = 0
        best_ind = 0


        for i in range(train_f.shape[0]):

            point = train_f[i]

            for j in range(train_f.shape[1]):

                dim_data = train_f[:,j]

                split_val = point[j]

                e1 =  (np.sum ( self.D * (data.train_l == self.pred_trans(dim_data > split_val)) ) ) / self.D_sum

                e2 =  (np.sum ( self.D * (data.train_l == self.pred_trans(dim_data <= split_val)) ) ) / self.D_sum

                better_e, better_dir = (e1, 1) if e1 < e2 else (e2, 0)

                if better_e < best_e :

                    best_e = better_e
                    best_dir = better_dir
                    best_dim = j
                    best_val = split_val
                    best_ind = i


        return best_val, best_dim, best_dir, best_e


    def update_D (self, data, weak_l, b_t):

        val, dim, dir = weak_l
       
        if dir == 1:
            preds = self.pred_trans(data.train_f[:,dim] > val)
        else:
            preds = self.pred_trans(data.train_f[:,dim] <= val)

        self.D *= np.exp(data.train_l * b_t * preds)
        self.D_sum = np.sum(self.D)

    
    def train(self, data):

        for i in range(self.num_epochs):
            print("Begin Train Epoch : {}/{}".format(i+1, self.num_epochs))

            val, dim, dir, e = self.best_weak(data)

            self.beta[i] = (np.log( (1-e))-np.log(e))/2

            self.weaks[i,0] = val
            self.weaks[i,1] = dim
            self.weaks[i,2] = dir

            self.update_D(data, (val, dim, dir), self.beta[i])

            print("Data in Epoch:= val:{}, dim:{}, dir:{}, e:{}".format(val, dim, dir, e))

            print("Train acc : {}, Test acc : {}".format(self.eval_model(data.train_f, data.train_l), self.eval_model(data.test_f, data.test_l)))

            print("-----------------------------------------")

    def ada_predict(self, sub_data):

        preds = np.zeros((sub_data.shape[0]))

        for i in range(self.num_epochs):

            val = self.weaks[i,0]
            dim = int(self.weaks[i,1])
            dir = self.weaks[i,2]

            if dir == 1:
                preds_i = self.pred_trans(sub_data[:,dim] > val)
            else:
                preds_i = self.pred_trans(sub_data[:,dim] <= val)

            preds += self.beta[i] * preds_i

        return self.pred_trans(preds>=0)

    def eval_model(self, sub_data, sub_l):

        preds = self.ada_predict(sub_data)

        return sum(preds == sub_l)/sub_data.shape[0]




def arg_parser():

    parser = argparse.ArgumentParser("Argument parser for AdaBoost")
    parser.add_argument("--train_root", type=str, default='../datasets/train_adaboost.csv', help="path to the training data")
    parser.add_argument("--test_root", type=str, default='../datasets/test_adaboost.csv', help="path to the test data")

    parser.add_argument("--num_epochs", type=int, default=10, help='number of epochs to train for')

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

    model = Ada_boost(data.train_f.shape, args.num_epochs)

    print("-----------------------------------------")
    print("Begin Training")
    print("-----------------------------------------")

    model.train(data)
    #import ipdb; ipdb.set_trace()
    print("-----------------------------------------")
    print("Finish Loading Data")
    print("-----------------------------------------")




