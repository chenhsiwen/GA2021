# genetic algorithm search of the one max optimization problem
from numpy.random import rand, randint, permutation, shuffle
import numpy as np
from util import *
import time

class SGA:
    def __init__(self, n_pop, n_bit, n_iter = 1000, s = 2, cross_type = '', r_cross = 0.5, r_mut = 0.1):
        self.n_pop = n_pop 
        self.n_bit = n_bit 
        self.n_iter = n_iter 
        self.s = s      
        self.cross_type = cross_type   
        self.r_cross = r_cross
        self.best_eval = 0
        self.best = 0
        self.f_t = 0
        self.sigma_t = 0
        self.I_t = 0
        self.I_ts = []

    def init_populations(self):
        self.populations = randint(2, size=(self.n_pop, self.n_bit))
    
    def set_populations(self, populations):
        self.populations = populations
        self.n_pop = populations.shape[0] 
        self.n_bit = populations.shape[1] 

    def fitness(self, chromosome):
        return onemax(chromosome)

    def eval(self):
        scores = [self.fitness(p) for p in self.populations]
        self.best = np.argmax(scores)
        self.best_eval = np.max(scores)
        f_t = np.mean(scores)
        sigma_t = np.std(scores)
        self.I_t = (f_t -self.f_t) / sigma_t
        self.f_t = f_t
        self.sigma_t = sigma_t
        self.I_ts.append(self.I_t)

    def selection(self):
        populations = self.populations.copy()
        perm = permutation(self.n_pop)
        populations = populations[perm]
        for _ in range(self.s - 1):
            _populations = populations[perm]
            populations = np.concatenate((populations, _populations), axis=0)
        selected = []
        for idx in range(self.n_pop):
            offset = self.s * idx
            tournaments = populations[offset:offset + self.s]
            scores = [self.fitness(p) for p in tournaments]
            best = np.argmax(scores)
            selected.append(tournaments[best])
        self.populations = np.array(selected)

    def onePointXO(self):
        polpulations = []
        for i in range(0, self.populations.shape[0], 2):
            childerns = onePointXO(self.populations[i], self.populations[i+1])
            polpulations = polpulations + childerns
        return np.array(polpulations)
    
    def uniformXO(self):
        polpulations = []
        for i in range(0, self.populations.shape[0], 2):
            childerns = uniformXO(self.populations[i], self.populations[i+1], self.r_cross)
            polpulations = polpulations + childerns
        return np.array(polpulations)

    def positionXO(self):
        populations = self.populations.copy()
        for i in range(self.n_bit):
            perm = permutation(self.n_pop)
            populations[:,i] = populations[perm,i]
        return populations

    def perfectXO(self):
        populations = self.populations.flatten()
        shuffle(populations)
        return populations.reshape(self.populations.shape)

    def crossover(self):
        if self.cross_type == 'uniform':
            self.populations = self.uniformXO()
        elif self.cross_type == 'position':
            self.populations = self.positionXO()
        elif self.cross_type == 'perfect':
            self.populations = self.perfectXO()
        elif self.cross_type == 'one':
            self.populations = self.onePointXO()

    def mutation(bitstring, r_mut):
        for i in range(self.n_pop):
            self.populations[i] = mutation(self.populations[i], r_mut)

    def generation(self):
        self.selection()     
        self.crossover()
        self.eval()
        return self.terminate()
        # self.mutation()
    
    def terminate(self):
        if (self.best_eval == self.n_bit):
            return 1
        return 0  
           
    def run(self):
        ts = time.time()
        self.init_populations()
        self.eval()
        for gen in range(self.n_iter):
            if (self.generation()):
                break
        te = time.time()
        return [self.n_bit, gen+1, te-ts]
    def intensity(self):
        ts = time.time()
        self.init_populations()
        self.eval()
        for gen in range(self.n_iter):
            if (self.generation()):
                break
        te = time.time()
        self.I_ts = self.I_ts[1:]
        return self.I_ts


        # return {'gen': gen + 1, 'best_eval': self.best_eval, 'time': te-ts, 'n_bit': self.n_bit, 'type': self.cross_type}
