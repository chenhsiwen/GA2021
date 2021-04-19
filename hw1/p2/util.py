
from numpy.random import rand, randint, permutation, shuffle
import pickle
import numpy as np
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

def onemax(x):
	return sum(x)

def onePointXO(p1,p2):
    idx = randint(0, p1.size)
    p1[idx:], p2[idx:] = p2.copy()[idx:], p1.copy()[idx:]
    return [p1, p2]

def uniformXO(p1, p2, r_cross):
    change = np.argwhere(rand(p1.size)<r_cross).flatten()
    p1[change], p2[change] = p2.copy()[change], p1.copy()[change]
    return [p1, p2]

def mutation(p1, r_mut):
    for i in range(p1.size):
        if rand() < r_mut:
            p1[i] = 1 -  p1[i]
    return p1

def save(data, name = 'hw2.pkl'):
    filename = os.path.join(dir_path, 'exp', name)
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)

def load(name = 'hw2.pkl'):
    filename = os.path.join(dir_path, 'exp', name)
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def MAPE(preds, y = 0.5762):
    preds = np.array(preds)
    return np.mean( np.abs(preds - y)/ y)

