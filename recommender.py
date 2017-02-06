import time

class Recommender(object):
    def __init__(self, ratings):
        self.n_user, self.n_item = ratings.shape
        self.n_rating = ratings.nnz
        self.ratings = ratings.copy()

    def update_params(self):
        raise NotImplementedError()

    def build(self):
        raise NotImplementedError()

    def predict(self, user, item):
        raise NotImplementedError()

    def calc_loss(self):
        raise NotImplementedError()

    def show_loss(self, itr, start_t, prev_loss):
        rap_t = time.time()
        cur_loss = self.calc_loss()
        msg = 'Iter={} [{}]\t [{}]loss: {} [{}]'
        print(msg.format(itr,
                         rap_t - start_t,
                         "-" if prev_loss >= cur_loss else "+",
                         cur_loss,
                         time.time() - rap_t))
        return cur_loss
