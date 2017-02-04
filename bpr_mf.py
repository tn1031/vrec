# coding: utf-8

import json
import sys
import time
import random
import numpy as np

from recommender import Recommender
from evaluator import Evaluator
from utils import *

class BPRmf(Recommender):
    def __init__(self, ratings, n_factor=10, reg=0.01, 
                 init_mean=0.0, init_stdev=0.1):
        super(BPRmf, self).__init__(ratings)

        # hyper parameters
        self.n_factor = n_factor  # number of latent factors.
        self.reg = reg  # regularization parameters
        self.init_mean = init_mean  # Gaussian mean for init V
        self.init_stdev = init_stdev  # Gaussian std-dev for init V

        # auxiliary variables
        self.U = np.random.normal(
            self.init_mean, self.init_stdev, self.n_user*self.n_factor).reshape((self.n_user, self.n_factor))
        self.V = np.random.normal(
            self.init_mean, self.init_stdev, self.n_item*self.n_factor).reshape((self.n_item, self.n_factor))

    def setUV(self, U, V):
        self.U = np.copy(U)
        self.V = np.copy(V)

    def build(self, maxiter=10, learn_rate=0.01, val_ratings=None, adaptive=False, show_total_loss=False):
        prev_loss = sys.float_info.max
        prev_hr = 0.0
        lr = learn_rate
        print("Run for BPRmf.")
        print(self.to_str())

        for itr in range(maxiter):
            start_t = time.time()

            loss = self.run_oneiter(lr)

            if show_total_loss:
                prev_loss = self.show_loss(itr, start_t, prev_loss)
            else:
                print('train loss:\t{} [{}]'.format(loss, time.time() - start_t))

            if adaptive and val_ratings is not None:
                evaluator = Evaluator(self, val_ratings)
                evaluator.evaluate(val_ratings)
                hr = np.mean(evaluator.ndcgs)
                lr = lr * 1.05 if hr > prev_hr else lr * 0.5
                prev_hr = hr

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

        grad_u = self.V[item, :] - self.V[j, :]
        self.U[user, :] += lr * (mult * grad_u - self.reg * self.U[user, :])

        grad = self.U[user, :]
        self.V[item, :] += lr * (mult * grad - self.reg * self.V[item, :])
        self.V[j, :] += lr * (-mult * grad - self.reg * self.V[j, :])

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

    def calc_loss(self):
        total_loss = self.reg * (np.sum(np.square(self.U)) + np.sum(np.square(self.V)))
        for u in range(self.n_user):
            loss = 0
            for i in self.ratings.getrowview(u).rows[0]:
                pred = self.predict(u, i)
                loss += np.power(self.ratings[u, i] - pred, 2)
            total_loss += loss

        return total_loss

    def predict(self, u, i):
        return np.dot(self.U[u], self.V[i])

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
        params = ['ratings', 'U', 'V']
        for p in params:
            d.pop(p)
        return json.dumps(d, indent=2)


if __name__ == '__main__':
    train, test = load_data('./yelp.rating')
    bpr = BPRmf(train)
    bpr.build(maxiter=5)

