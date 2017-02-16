import numpy as np
import json
import random
import sys
import time
#sys.path.append('~/python/pypropack')
#from pypropack import svdp
from scipy.sparse.linalg import svds

from recommender import Recommender


class LoCo(Recommender):
    def __init__(self, ratings, sideinfo, n_factor=10, reg=0.1):
        super(LoCo, self).__init__(ratings)

        # side information
        self.sideinfo = sideinfo.copy()

        # auxiliary variables
        self.Z = np.random.uniform(low=-.001, high=.001, size=(self.n_item, n_factor))

        # hyper parameters
        self.n_factor = n_factor
        self.reg = reg

        # svd
        u, s, v = svds(self.sideinfo, n_factor)
        self.V = v   # n_factor x n_attr


    def update_params(self, user, i, lr):
        # sample a negative item(uniformly random)
        j = np.random.randint(self.n_item)
        while self.ratings[user, j] != 0:
            j = np.random.randint(self.n_item)

        # BPR update rules
        y_pos = self.predict(user, i)  # target value of positive instance
        y_neg = self.predict(user, j)  # target value of negative instance
        mult = self.partial_loss(y_pos - y_neg)

        grad = np.dot(self.sideinfo[user, :].toarray(), self.V.T).flatten()
        self.Z[i, :] += lr * ( mult * grad - self.reg * self.Z[i, :])
        self.Z[j, :] += lr * (-mult * grad - self.reg * self.Z[j, :])

        return (y_pos - y_neg)**2


    def run_onriter(self, learn_rate):
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

    def partial_loss(self, x):
        exp_x = np.exp(-x)
        return exp_x / (1.0 + exp_x)

    def predict(self, user, item):
        return np.dot(self.sideinfo[user, :].toarray(),
                      np.dot(self.V.T, self.Z[item, :].T)).flatten()[0]

    def build(self, maxiter=10, learn_rate=0.01, show_total_loss=False):
        print('build LoCo')
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
        params = ['ratings', 'sideinfo', 'Z', 'V']
        for p in params:
            d.pop(p)
        return json.dumps(d, indent=2)


if __name__ == '__main__':
    from evaluator import Evaluator
    from utils import load_movielens_inv

    train, test, attr = load_movielens_inv()
    model = LoCo(train, attr, n_factor=6)
    model.build(maxiter=5)

    ev = Evaluator(model, test[:20000])
    ev.evaluate_online(interval=10)
