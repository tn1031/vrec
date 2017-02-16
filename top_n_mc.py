import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot
#from pypropack import svdp
from scipy.sparse.linalg import svds

from utils import load_data


def update_X(X, mu, k=6):
    #U, S, VT = svdp(X, k=k)
    U, S, VT = svds(X, k=k, which='LM')
    P = np.c_[np.ones((k, 1)), 1-S, 1./2./mu-S]
    sigma_star = np.zeros(k)
    for t in range(k):
        p = P[t, :]
        delta = p[1]**2 - 4 * p[0] * p[2]
        if delta <= 0:
            sigma_star[t] = 0.
        else:
            solution = np.roots(p)
            solution = solution.tolist()
            solution.sort(key=abs)
            solution = np.array(solution)
            if solution[0] * solution[1] <= 0:
                sigma_star[t] = solution[1]
            elif solution[1] < 0:
                sigma_star[t] = 0.
            else:
                f = np.log(1 + solution[1]) + mu * (solution[1] - s[t])**2
                if f > mu *s[t]**2:
                    sigma_star[t] = 0.
                else:
                    sigma_star[t] = solution[1]

    sigma_star = sp.csr_matrix(np.diag(sigma_star))
    sigma_star = safe_sparse_dot(safe_sparse_dot(U, sigma_star), VT)
    sigma_star[abs(sigma_star)<1e-10] = 0
    return sp.lil_matrix(sigma_star)

def maximum (A, B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

def mc_logdet(train, mu=1., gamma=5, maxitr=2):
    m, n = train.shape
    nonzero_row, nonzero_col = train.nonzero()
    nonzero_index = zip(nonzero_row, nonzero_col)
    prevX = train#.toarray()
    X = None
    Y = train#.toarray()
    Z = sp.lil_matrix(np.zeros((m, n)))
    
    for itr in range(maxitr):
        X = update_X(Y - Z / mu, mu / 2., 6)

        for idx in nonzero_index:
            X[idx[0], idx[1]] = train[idx[0], idx[1]]
        Y = maximum(X + (Z/mu), sp.lil_matrix(np.zeros(X.shape)))
        Z = Z + mu * (X - Y)
        mu *= gamma

        #err = np.sum((X-prevX)**2)/np.sum(prevX**2)
        #print err

        prevX = X

    return X


if __name__ == '__main__':
    train, test = load_data('./../bpr/yelp.rating')
    mc_logdet(train)
