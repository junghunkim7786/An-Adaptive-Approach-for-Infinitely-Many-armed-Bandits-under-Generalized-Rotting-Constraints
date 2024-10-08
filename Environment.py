import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from scipy.stats import bernoulli
from scipy import stats


class rotting_many_Env: #slow rotting
    def inverse_cdf(self, y, beta):
    # Computed analytically
        return 1-(1-y)**(1/beta)

    def sample_distribution(self,beta):
        uniform_random_sample = random.random()
        return self.inverse_cdf(uniform_random_sample,beta)

    def __init__(self,rho,seed,T,beta=1):
        np.random.seed(seed)
        self.optimal=1
        self.exp_reward=np.zeros(T)
        self.rho=rho
        self.T=T
        if beta==1:
            for k in range(self.T):
                self.exp_reward[k]=np.random.uniform(0,1)
        else: 
            for k in range(self.T):
                self.exp_reward[k]=self.sample_distribution(beta)



    def observe(self,k,t):
        reward=self.exp_reward[k]+np.random.normal(0,1)
        exp_reward=self.exp_reward[k]
        rho_t=self.rho*(1/(t+1))
        self.exp_reward[k]=self.exp_reward[k]-rho_t
        return exp_reward, reward
    

    