# coding: utf-8

import json
import sys
import time
import random
import numpy as np

from recommender import Recommender
from evaluator import Evaluator
from utils import *

class BPRknn(Recommender):
    def __init__(self, ratings,
                 n_factor=10, reg_pos=0.01, reg_neg=0.01,
                 init_mean=0.0, init_stdev=0.1):
        super(BPRknn, self).__init__(ratings)

        # hyper parameters
        self.reg_pos = reg_pos  # regularization parameters
        self.reg_neg = reg_neg  # regularization parameters
        self.init_mean = init_mean  # Gaussian mean for init V
        self.init_stdev = init_stdev  # Gaussian std-dev for init V

        # auxiliary variables
        self.C = np.random.normal(
            self.init_mean, self.init_stdev, self.n_item*self.n_item).reshape((self.n_item, self.n_item))

    def setUV(self, C):
        self.C = np.copy(C)

    def build(self, maxiter=10, learn_rate=0.01, val_ratings=None, adaptive=False, show_total_loss=False):
        loss_prev = sys.float_info.max
        hr_prev = 0.0
        lr = learn_rate
        print("Run for BPRmf.")
        print(self.to_str())

        for itr in range(maxiter):
            start_t = time.time()

            loss = self.run_oneiter(lr)

            if show_total_loss:
                loss_prev = self.show_loss(itr, start_t, loss_prev)
            else:
                print('train loss:\t{} [{}]'.format(loss, time.time() - start_t))

            if adaptive and val_ratings is not None:
                evaluator = Evaluator(self, val_ratings)
                evaluator.evaluate(val_ratings)
                hr = np.mean(evaluator.ndcgs)
                lr = lr * 1.05 if hr > hr_prev else lr * 0.5
                hr_prev = hr

    # Run model for one iteration
    def run_oneiter(self, learn_rate):
        loss = 0.0
        # Each training epoch
        for s in range(self.n_rating):
            u = np.random.randint(self.n_user)
            item_list = self.ratings.getrowview(u).rows[0]
            if len(item_list) == 0:
                continue

            # sample a positibe item
            i = random.choice(item_list)

            # One SGD update
            loss += self.update_params(u, i, learn_rate)

        return loss / self.n_rating

    def update_params(self, user, item, lr):
        # sample a negative item(uniformly random)
        j = np.random.randint(self.n_item)
        while self.ratings[user, j] != 0:
            j = np.random.randint(self.n_item)

        # BPR update rules
        y_pos = self.predict(user, item)  # target value of positive instance
        y_neg = self.predict(user, j)  # target value of negative instance
        mult = self.partial_loss(y_pos - y_neg)

        item_list = self.ratings.getrow(user).rows[0]

        # update {c_il, c_li}
        if item in item_list:
            item_list.remove(item)
        if len(item_list) > 0:
            self.C[item, np.array(item_list)] += lr * (mult - self.reg_pos * self.C[item, np.array(item_list)])
            self.C[np.array(item_list), item] += lr * (mult - self.reg_pos * self.C[np.array(item_list), item])

        item_list.append(item)
        if j in item_list:
            item_list.remove(j)
        if len(item_list) > 0:
            self.C[j, np.array(item_list)] += lr * (-mult - self.reg_neg * self.C[j, np.array(item_list)])
            self.C[np.array(item_list), j] += lr * (-mult - self.reg_neg * self.C[np.array(item_list), j])

        return (y_pos - y_neg)**2

    # Partial of the ln sigmoid function used by BPR
    def partial_loss(self, x):
        exp_x = np.exp(-x)
        return exp_x / (1.0 + exp_x)

    def _showProgress(self, itr, start_t, val_ratings):
        end_itr = time.time()
        evaluator = Evaluator(self, val_ratings)
        if self.n_user == len(val_ratings):
            # leave-1-out eval
            evaluator.evaluate(val_ratings)
        else:
            # global split
            evaluator.evaluateOnline(val_ratings, 1000)
        end_eval = time.time()
        
        sys.stderr.write(
            "Iter={}[{}] <loss, hr, ndcg, prec>:\t {}\t {}\t {}\t {}\t [{}]\n".format(itr, 
                                                                                     end_itr - start_t, self.calc_loss(), 
                                                                                     np.mean(evaluator.hits), np.mean(evaluator.ndcgs), np.mean(evaluator.precs), 
                                                                                     end_eval - end_itr))

    #  Fast way to calculate the loss function
    def calc_loss(self):
        total_loss = 0.0
        for u in range(self.n_user):
            loss = 0
            for i in self.ratings.getrowview(u).rows[0]:
                pred = 1.0 / (1.0 + np.exp(-self.predict(u, i)))   # sigmoid
                loss += np.power(self.ratings[u, i] - pred, 2)
            total_loss += loss

        return total_loss

    def predict(self, user, item):
        item_list = self.ratings.getrow(user).rows[0]
        if item in item_list:
            item_list.remove(item)
            
        if len(item_list) == 0:
                return 0.0

        x = np.sum(self.C[item, np.array(item_list)])
        return x

    def updateModel(self, user, item):
        self.ratings[user, item] = 1

        # user retain
        item_list = self.ratings.getrowview(user).rows[0]
        for itr in range(self.maxIterOnline):
            random.shuffle(item_list)

            for s in range(len(item_list)):
                # retrain for the user or for the (user, item) pair
                i = item_list[s] if self.onlineMode == 'u' else item
                self.update_ui(u, i)

    def to_str(self):
        d = self.__dict__.copy()
        params = ['ratings', 'C']
        for p in params:
            d.pop(p)
        return json.dumps(d, indent=2)


if __name__ == '__main__':
    train, test = load_data('./yelp.rating')
    bpr = BPRknn(train)
    bpr.build(maxiter=5)

