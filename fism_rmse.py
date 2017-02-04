import numpy as np
import json
import random
import sys
import time

from recommender import Recommender
from evaluator import Evaluator
from utils import load_data


class FISMrmse(Recommender):
    def __init__(self, ratings, n_factor=10, rho=0.5, alpha=0.5, beta=0.1, reg_user=1.0, reg_item=1.0):
        super(FISMrmse, self).__init__(ratings)

        # auxiliary variables
        self.bias_u = np.zeros(self.n_user)
        self.bias_i = np.zeros(self.n_item)
        self.P = np.random.uniform(low=-.001, high=.001, size=(self.n_item, n_factor))
        self.Q = np.random.uniform(low=-.001, high=.001, size=(self.n_item, n_factor))

        # hyper parameters
        self.n_factor = n_factor
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.reg_user = reg_user
        self.reg_item = reg_item

    def update_params(self, user, item, rating, lr):
        x = np.zeros(self.n_factor)
        item_list = self.ratings.getrow(user).rows[0]
        if len(item_list) == 0:
            return 0.0

        if item in item_list:
            item_list.remove(item)
        n_u_plus = len(item_list)
        if n_u_plus == 0:
            matched_degree = 0
        else:
            matched_degree = np.power(n_u_plus, -self.alpha)
            x = np.sum(self.P[np.array(item_list), :], axis=0)
            x *= matched_degree

        r_ui = self.bias_u[user] + self.bias_i[item] + np.dot(x, self.Q[item, :])
        e_ui = rating - r_ui
        loss = e_ui**2

        self.bias_u[user] += lr * (e_ui - self.reg_user * self.bias_u[user])
        self.bias_i[item] += lr * (e_ui - self.reg_item * self.bias_i[item])
        self.Q[item, :] += lr * (e_ui * x - self.beta * self.Q[item, :])
        if n_u_plus > 0:
            self.P[np.array(item_list), :] += lr * (e_ui * matched_degree * self.Q[item, :] - self.beta * self.P[np.array(item_list), :])

        return loss

    def run_onriter(self, learn_rate):
        neg_set_size = int(self.rho * self.n_rating)
        nonzero_row, nonzero_col = self.ratings.nonzero()
        pos_set = set(zip(nonzero_row, nonzero_col))

        # sample negative ratings
        seq_u = np.random.randint(self.n_user, size=neg_set_size)
        seq_i = np.random.randint(self.n_item, size=neg_set_size)

        train_set = list(pos_set.union(set(zip(seq_u, seq_i))))
        random.shuffle(train_set)

        loss = 0.0
        for u, i in train_set:
            loss += self.update_params(u, i, self.ratings[u, i], learn_rate)

        return loss / len(train_set)

    def predict(self, user, item):
        x = np.zeros(self.n_factor)
        item_list = self.ratings.getrow(user).rows[0]
        if len(item_list) == 0:
            return 0.0

        if item in item_list:
            item_list.remove(item)
        n_u_plus = len(item_list)
        if n_u_plus > 0:
            matched_degree = np.power(n_u_plus, -self.alpha)
            x = np.sum(self.P[np.array(item_list), :], axis=0)
            x *= matched_degree

        r_ui = self.bias_u[user] + self.bias_i[item] + np.dot(x, self.Q[item, :])
        return r_ui

    def build(self, maxiter=10, learn_rate=0.01, show_total_loss=False):
        print('build FISMrmse')
        print(self.to_str())
        prev_loss = sys.float_info.max

        # SGD
        for itr in range(maxiter):
            start_t = time.time()
            loss = self.run_onriter(learn_rate)

            if show_total_loss:
                prev_loss = self.show_loss(itr, start_t, prev_loss)
            else:
                print('train loss:\t{} [{}]'.format(loss, time.time() - start_t))

    def calc_loss(self):
        total_loss = self.beta * (np.sum(np.square(self.P)) + np.sum(np.square(self.Q)))
        total_loss += self.reg_user * np.sum(np.square(self.bias_u))
        total_loss += self.reg_item * np.sum(np.square(self.bias_i))

        for u in range(self.n_user):
            loss = 0.0
            item_list = self.ratings.getrow(u).rows[0]
            n_u_plus = len(item_list)
            if n_u_plus == 0:
                continue
            matched_degree = np.power(n_u_plus - 1, -self.alpha) if n_u_plus > 1 else 0
            sum_p = np.sum(self.P[np.array(item_list), :], axis=0)

            for i in item_list:
                t = (sum_p[:] - self.P[i, :]) * matched_degree

                r_ui = self.bias_u[u] + self.bias_i[i] + np.dot(t, self.Q[i, :])
                e_ui = self.ratings[u, i] - r_ui
                loss = e_ui**2
    
            total_loss += loss

        return total_loss

    def to_str(self):
        d = self.__dict__.copy()
        params = ['ratings', 'bias_u', 'bias_i', 'P', 'Q']
        for p in params:
            d.pop(p)
        return json.dumps(d, indent=2)


if __name__ == '__main__':
    train, test = load_data('./yelp.rating')
    fism = FISMrmse(train, rho=0.2)
    fism.build(maxiter=5)
