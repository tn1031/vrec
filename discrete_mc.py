import json
import random
import time
import numpy as np

from recommender import Recommender

class DiscreteMC(Recommender):
    def __init__(self, ratings, n_factor, score_max, gamma=0.1, eta=0.1):
        super(DiscreteMC, self).__init__(ratings)

        self.n_factor = n_factor
        self.U = np.random.uniform(low=-.001, high=.001, size=(self.n_user, n_factor))
        self.V = np.random.uniform(low=-.001, high=.001, size=(self.n_item, n_factor))
        self.threshold_d = np.random.uniform(low=0, high=1, size=score_max+1)

        self.gamma = gamma
        self.eta = eta

    def update_params(self, u, i, lr):
        pred = self.predict(u, i)
        rating = self.ratings[u, i]
        t = int(rating)
        
        first = max(pred - rating - self.threshold_d[t], 0)
        second = max(-pred + rating - 1 + self.threshold_d[t-1], 0)

        # update U
        grad = first * self.V[i] - second * self.V[i] + self.gamma * self.U[u]
        self.U[u, :] -= lr * grad

        # update V
        grad = first * self.U[u] - second * self.U[u] + self.gamma * self.V[i]
        self.V[i, :] -= lr * grad

        # update threshold
        first = max(pred - rating - self.threshold_d[t], 0)
        grad = -first + self.eta * (self.threshold_d[t] - 0.5)
        self.threshold_d[t] -= lr * grad
        self.threshold_d[t] = min(max(self.threshold_d[t], 0), 1)

        first = max(pred - rating - 1 + self.threshold_d[t-1], 0)
        grad = first + self.eta * (self.threshold_d[t-1] - 0.5)
        self.threshold_d[t-1] -= lr * grad
        self.threshold_d[t-1] = min(max(self.threshold_d[t-1], 0), 1)

        return (rating - pred)**2

    def one_iter(self, learn_rate):
        nonzero_row, nonzero_col = self.ratings.nonzero()
        train_set = zip(nonzero_row, nonzero_col)
        random.shuffle(train_set)

        loss = 0.0
        for u, i in train_set:
            loss += self.update_params(u, i, learn_rate)

        return loss

    def build(self, maxiter=10, learn_rate=0.01):
        print('build Discrete Matrix Completion')
        print(self.to_str())

        # SGD
        for itr in range(maxiter):
            start_t = time.time()
            loss = self.one_iter(learn_rate)

            print('train loss:\t{} [{}]'.format(loss, time.time() - start_t))

    def predict(self, user, item):
        return np.dot(self.U[user], self.V[item])

    def calc_loss(self):
        total_loss = self.gamma * (np.sum(np.square(self.U)) + np.sum(np.square(self.V)))

        nonzero_row, nonzero_col = self.ratings.nonzero()
        train_set = zip(nonzero_row, nonzero_col)
        for u, i in train_set:
            pred = self.predict(u, i)
            rating = self.ratings[u, i]
            t = int(rating)
            total_loss += max(pred - rating - self.threshold_d[t], 0)**2
            total_loss += max(-pred + rating - 1 + self.threshold_d[t-1], 0)**2
            total_loss += self.eta * (self.threshold_d[t] - 0.5)**2
            total_loss += self.eta * (self.threshold_d[t-1] - 0.5)**2

        return total_loss

    def to_str(self):
        d = self.__dict__.copy()
        params = ['ratings', 'threshold_d', 'U', 'V']
        for p in params:
            d.pop(p)
        return json.dumps(d, indent=2)


if __name__ == "__main__":
    from utils import load_data
    from evaluator import Evaluator

    train, test = load_data("./../bpr/yelp.rating")
    model = DiscreteMC(train, 300, 1)

    model.build(maxiter=500, learn_rate=0.05)
    evaluator = Evaluator(model, test[:10000])
    evaluator.evaluate_online(interval=10)

    model.build(maxiter=500, learn_rate=0.02)
    evaluator = Evaluator(model, test[:10000])
    evaluator.evaluate_online(interval=10)

    model.build(maxiter=2000, learn_rate=0.01)
    evaluator = Evaluator(model, test[:10000])
    evaluator.evaluate_online(interval=10)
