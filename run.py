
from bpr_mf import BPRmf
from bpr_knn import BPRknn
from fism_rmse import FISMrmse
from fism_auc import FISMauc
from evaluator import Evaluator
from utils import *


if __name__ == "__main__":
    train, test = load_data('./yelp.rating')

    bprmf = BPRmf(train)
    bprmf.build(maxiter=7)
    evaluator = Evaluator(bprmf, test[:10000])
    evaluator.evaluate_online(interval=10)

    bprknn = BPRknn(train)
    bprknn.build(maxiter=7)
    evaluator = Evaluator(bprknn, test[:10000])
    evaluator.evaluate_online(interval=10)

    fismrmse = FISMrmse(train, rho=0.3)
    fismrmse.build(maxiter=7)
    evaluator = Evaluator(fismrmse, test[:10000])
    evaluator.evaluate_online(interval=10)

    fismauc = FISMauc(train, rho=10)
    fismauc.build(maxiter=7)
    evaluator = Evaluator(fismauc, test[:10000])
    evaluator.evaluate_online(interval=10)
