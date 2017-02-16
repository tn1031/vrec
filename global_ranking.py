import numpy as np                                     
                                                       
from recommender import Recommender                    
from evaluator import Evaluator                        
from utils import load_data                            
                                                       
                                                       
class GlobalRank(Recommender):                         
    def __init__(self, ratings):                       
        super(GlobalRank, self).__init__(ratings)      
        self.ranking = np.sum(ratings.toarray(), axis=0)
        self.ranking /= np.sum(ratings.toarray())      
                                                       
    def update_params(self, user, item, rating, lr):   
        pass                                           
                                                       
    def run_oneiter(self, learn_rate):                 
        pass                                           
                                                       
    def predict(self, user, item):                     
        return self.ranking[item]                      
                                                       
    def build(self):                                   
        pass                                           
                                                       
    def calc_loss(self):                               
        pass                                           
                                                       
    def to_str(self):                                  
        pass                                           
                                                       
                                                       
if __name__ == '__main__':                             
    train, test = load_data('./yelp.rating')           
    model = GlobalRank(train)                          
                                                       
    evaluator = Evaluator(model, test[:10000])         
    evaluator.evaluate_online(interval=10)             