import json
import sys
import time

import numpy as np
from pypropack import svdp
from sklearn.utils.extmath import safe_sparse_dot

from recommender import Recommender


class LogdetMC(Recommender):
    def __init__(self, ratings, n_factor=20, mu_0=1.0, gamma=3.0):
        super(LogdetMC, self).__init__(ratings)

        self.n_factor = n_factor
        nonzero_row, nonzero_col = train.nonzero()
        self.nonzero_index = zip(nonzero_row, nonzero_col)

        M = self.ratings.toarray()
        self.M = np.asarray(M)
        self.X = np.asarray(M)
        self.Y = np.asarray(M)
        self.Z = np.zeros((self.n_user, self.n_item))
        self.mu_0 = mu_0
        self.mu = mu_0
        self.gamma = gamma

    def update_X(self, X, mu, k=20):
        U, S, VT = svdp(X, k=k)
        P = np.c_[np.ones((k, 1)), 1-S, 1./2./mu-S]
        sigma_star = np.zeros(k)
        for t in range(k):
            p = P[t, :]
            delta = p[1]**2 - 4 * p[0] * p[2]
            if delta <= 0:
                sigma_star[t] = 0.
            else:
                solution = np.roots(p)
                solution = sorted(solution, key=abs)
                solution = np.array(solution)
                if solution[0] * solution[1] <= 0:
                    sigma_star[t] = solution[1]
                elif solution[1] < 0:
                    sigma_star[t] = 0.
                else:
                    f = np.log(1 + solution[1]) + mu * (solution[1] - s[t])**2
                    if f > mu * s[t]**2:
                        sigma_star[t] = 0.
                    else:
                        sigma_star[t] = solution[1]

        sigma_star = np.diag(sigma_star)
        sigma_star = np.dot(np.dot(U, sigma_star), VT)
        return sigma_star

    def update_params(self):
        pass

    def one_iter(self):
        self.X = self.update_X(self.Y - self.Z / self.mu,
                               self.mu / 2., self.n_factor)
        for idx in self.nonzero_index:
            self.X[idx[0], idx[1]] = self.M[idx[0], idx[1]]
        self.Y = np.maximum(self.X + (self.Z / self.mu),
                            np.zeros((self.n_user, self.n_item)))
        self.Z = self.Z + self.mu * (self.X - self.Y)
        self.mu *= self.gamma

    def build(self, maxiter=10):
        for itr in range(maxiter):
            self.one_iter()

    def predict(self, user, item):
        return self.X[user, item]

    def calc_loss(self):
        return np.sum((self.M-self.X)**2)/np.sum(self.M**2)

    def to_str(self):
        d= self.__dict__.copy()
        params = ['ratings', 'X', 'Y', 'Z', 'M']
        for p in params:
            d.pop(p)
        return json.dumps(d, indent=2)


if __name__ == '__main__':
    from utils import load_data
    from evaluator import Evaluator

    train, test = load_data('./yelp.rating')
    model = LogdetMC(train)

    model.build(maxiter=5)

    evaluator = Evaluator(model, test[:10000])
    evaluator.evaluate_online(interval=10)


