import numpy as np
import json
import random
import sys
import time

from recommender import Recommender
from evaluator import Evaluator
from utils import load_data


class FISMauc(Recommender):
    def __init__(self, ratings, n_factor=10, alpha=0.5, beta=1.0, rho=20, reg_item=1.0):
        super(FISMauc, self).__init__(ratings)

        # auxiliary variables
        self.bias_i = np.zeros(self.n_item)
        self.P = np.random.uniform(low=-.001, high=.001, size=(self.n_item, n_factor))
        self.Q = np.random.uniform(low=-.001, high=.001, size=(self.n_item, n_factor))

        # hyper parameters
        self.n_factor = n_factor
        self.alpha = alpha
        self.rho = rho    # negative sample size
        self.beta = beta    # weight of P,Q
        self.reg_item = reg_item    # regularize item bias

    def update_params(self):
        pass

    
    def run_oneiter(self, learn_rate):
        loss = 0.0
        for u in range(self.n_user):
            item_list = self.ratings.getrow(u).rows[0]
            n_u_plus = len(item_list)
            if n_u_plus == 0:
                continue
            matched_degree = np.power(n_u_plus - 1, -self.alpha) if n_u_plus > 1 else 0

            sum_p = np.sum(self.P[np.array(item_list), :], axis=0)

            for i in item_list:
                rating = self.ratings[u, i]
                x = np.zeros(self.n_factor)
                t = (sum_p[:] - self.P[i, :]) * matched_degree

                Z = set()
                while len(Z) < self.rho:
                    j = np.random.randint(self.n_item)
                    while self.ratings[u, j] != 0:
                        j = np.random.randint(self.n_item)
                    Z.add(j)

                for j in Z:
                    r_ui = self.bias_i[i] + np.dot(t, self.Q[i, :])
                    r_uj = self.bias_i[j] + np.dot(t, self.Q[j, :])
                    e_ui = rating - (r_ui - r_uj)

                    loss += np.sum(e_ui**2)

                    self.bias_i[i] += learn_rate * ( e_ui - self.reg_item * self.bias_i[i])
                    self.bias_i[j] += learn_rate * (-e_ui - self.reg_item * self.bias_i[j])

                    x += e_ui * (self.Q[i, :] - self.Q[j, :])
                    self.Q[i, :] += learn_rate * ( e_ui * t - self.beta * self.Q[i, :])
                    self.Q[j, :] += learn_rate * (-e_ui * t - self.beta * self.Q[j, :])

                item_list.remove(i)
                if len(item_list) > 0:
                    deri = matched_degree / self.rho
                    self.P[np.array([item_list]), :] += learn_rate * (deri * x - self.beta * self.P[np.array([item_list]), :])

        return loss / self.n_user

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

        r_ui = self.bias_i[item] + np.dot(x, self.Q[item, :])
        return r_ui

    def build(self, maxiter=10, learn_rate=0.01, show_total_loss=False):
        print('build FISMauc')
        print(self.to_str())
        prev_loss = sys.float_info.max

        # SGD
        for itr in range(maxiter):
            start_t = time.time()
            loss = self.run_oneiter(learn_rate)

            if show_total_loss:
                prev_loss = self.show_loss(itr, start_t, prev_loss)
            else:
                print('train loss:\t{} [{}]'.format(loss, time.time() - start_t))

    def calc_loss(self):
        total_loss = self.beta * (np.sum(np.square(self.P)) + np.sum(np.square(self.Q)))
        total_loss += self.reg_item * np.sum(np.square(self.bias_i))
        for u in range(self.n_user):
            loss = 0
            item_list = self.ratings.getrow(u).rows[0]
            n_u_plus = len(item_list)
            if n_u_plus == 0:
                continue
            matched_degree = np.power(n_u_plus - 1, -self.alpha) if n_u_plus > 1 else 0
            sum_p = np.sum(self.P[np.array(item_list), :], axis=0)
            
            for i in item_list:
                rating = self.ratings[u, i]
                t = (sum_p[:] - self.P[i, :]) * matched_degree

                Z = set()
                while len(Z) < self.rho:
                    j = np.random.randint(self.n_item)
                    while self.ratings[u, j] != 0:
                        j = np.random.randint(self.n_item)
                    Z.add(j)

                for j in Z:
                    r_ui = self.bias_i[i] + np.dot(t, self.Q[i, :])
                    r_uj = self.bias_i[j] + np.dot(t, self.Q[j, :])
                    e_ui = rating - (r_ui - r_uj)

                    loss += np.sum(e_ui**2)

            total_loss += loss

        return total_loss


    def to_str(self):
        d = self.__dict__.copy()
        params = ['ratings', 'bias_i', 'P', 'Q']
        for p in params:
            d.pop(p)
        return json.dumps(d, indent=2)


if __name__ == '__main__':
    train, test = load_data('./yelp.rating')
    fism = FISMauc(train, rho=20)
    fism.build(maxiter=5)
