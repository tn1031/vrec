import numpy as np
from utils import *

class Evaluator():
    def __init__(self, model, ratings):
        self.model = model
        self.ratings = ratings
        self.hits = None
        self.ndcgs = None
        self.precs = None

    def evaluate(self):
        self.hits = np.array([0.0] * self.model.n_user)
        self.ndcgs = np.array([0.0] * self.model.n_user)
        self.precs = np.array([0.0] * self.model.n_user)

        for rating in self.ratings:
            u = rating[0]
            i = rating[1]
            hit, ndcg, prec = self.evaluate_for_user(u, i)
            self.hits[u] = hit
            self.ndcgs[u] = ndcg
            self.precs[u] = prec

    def evaluate_online(self, interval=10):
        test_count = len(self.ratings)
        self.hits = np.array([0.0] * test_count)
        self.ndcgs = np.array([0.0] * test_count)
        self.precs = np.array([0.0] * test_count)

        counts = [0] * (interval + 1)
        hits_r = [0.0] * (interval + 1)
        ndcgs_r = [0.0] * (interval + 1)
        precs_r = [0.0] * (interval + 1)

        #update_time = 0
        for t, rating in enumerate(self.ratings):
            if t > 0 and t % 500 == 0:
                msg = '{}: <ht, ndcg, prec> =\t {}\t {}\t {}'
                print(msg.format(t,
                                 np.sum(self.hits) / t,
                                 np.sum(self.ndcgs) / t,
                                 np.sum(self.precs) / t))

            hit, ndcg, prec = self.evaluate_for_user(rating[0], rating[1])
            self.hits[t] = hit
            self.ndcgs[t] = ndcg
            self.precs[t] = prec

            # statistics for break down
            r = len(self.model.ratings.getrowview(rating[0]).rows[0])
            r = interval if r > interval else r
            counts[r] += 1
            hits_r[r] += hit
            ndcgs_r[r] += ndcg
            precs_r[r] += prec

            # update the model (for sequential update)
            #start_t = time.time()
            #self.model.update_factors(rating[0], rating[1], 1, 0.01)
            #update_time += time.time() - start_t

        print('Break down the results by number of user ratings for the test pair.')
        print('#Rating\t Percentage\t HR\t NDCG\t MAP\n')
        for i in range(interval + 1):
            if counts[i] == 0:
                continue
            msg = '{}\t {}%%\t {}\t {}\t {}'
            print(msg.format(i,
                             float(counts[i]) / test_count * 100,
                             hits_r[i] / counts[i],
                             ndcgs_r[i] / counts[i],
                             precs_r[i] / counts[i]))

        #print('Avg model update time per instance: {}'.format(float(update_time)/test_count))

    def evaluate_for_user(self, user, item, topK=100, ignore_train=True):
        map_item_score = {}
        # Get the score of the test item first.
        max_score = self.model.predict(user, item)

        # Early stopping if there are topK items larger than maxScore.
        count_larger = 0
        for i in range(self.model.n_item):
            score = self.model.predict(user, i)
            map_item_score[i] = score

            if score > max_score:
                count_larger += 1
            if count_larger > topK:
                # early stopping
                return 0.0, 0.0, 0.0

        # Selecting topK items (does not exclude train items).
        if ignore_train:
            rated_item = self.model.ratings.getrowview(user).rows[0]
            rank_list = TopKeysByValue(map_item_score, topK, rated_item)
        else:
            rank_list = TopKeysByValue(map_item_score, topK, None)

        hit = self._calc_hr(rank_list, item)
        ndcg = self._calc_ndcg(rank_list, item)
        prec = self._calc_precision(rank_list, item)

        return hit, ndcg, prec

    def _calc_hr(self, rank_list, gt_item):
        for item in rank_list:
            if item == gt_item:
                return 1
        return 0

    def _calc_ndcg(self, rank_list, gt_item):
        for i, item in enumerate(rank_list):
            if item == gt_item:
                return np.log(2) / np.log(i + 2)
        return 0

    def _calc_precision(self, rank_list, gt_item):
        for i, item in enumerate(rank_list):
            if item == gt_item:
                return 1.0 / (i + 1)
        return 0
