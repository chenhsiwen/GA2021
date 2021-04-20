from SGA import SGA
import numpy as np
import math
from multiprocessing import Pool
from util import *
import time
from itertools import product

cross_types = [ 'one', 'uniform', 'position', 'perfect']
ell = 10
n_bits = [50 * (i + 1) for i in range(ell)]
num_exp = 30

def job(task):
    [cross_type, n_bit] = task
    row = []
    ts = time.time()
    for _ in range(num_exp):
        n_pop = 4 * int(n_bit * math.log(n_bit))
        ga = SGA(n_bit=n_bit, n_pop=n_pop, cross_type=cross_type)
        col = ga.run()
        row.append(col)
    te = time.time()
    avg = np.mean(row, axis = 0)
    print(cross_type, n_bit, te-ts, avg[:3])
    return avg

def pretty_print(exp):
    for i, results in enumerate(exp):
        print('=========== {} ==========='.format(cross_types[i]))
        for r in results:
            print(r)


if __name__ == '__main__':
    ts = time.time()
    tasks = list(product(cross_types, n_bits))
    with Pool(4) as p:
        exp = list(p.map(job, tasks))
        p.close()
    exp = np.array(exp).reshape(len(cross_types), len(n_bits), len(exp[0]))
    save(exp, 'covergence30.pkl')
    exp = load('covergence30.pkl')
    pretty_print(exp)
    print(exp)
    te = time.time()
    print('time: {}'.format(te-ts))

