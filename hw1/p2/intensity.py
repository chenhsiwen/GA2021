from SGA import SGA
import numpy as np
import math
from multiprocessing import Pool
from util import *
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

cross_types = [ 'one', 'uniform', 'population', 'perfect']
n_bits = [50, 100, 200, 400]
num_exp = 10

def job(cross_type):
    results = []
    for n_bit in n_bits:
        row = []
        for _ in range(num_exp):
            n_pop = 4 * int(n_bit * math.log(n_bit))
            ga = SGA(n_bit=n_bit, n_pop=n_pop, cross_type=cross_type)
            r = ga.intensity()
            row.append(r)
        size = np.max([len(r) for r in row])
        count = np.full(size, 0)
        acc = np.full(size, 0.0)
        for r in row:
            for i, c in enumerate(r):
                count[i] += 1
                acc[i] += c
        result = acc/count
        results.append(result)
        print(cross_type, n_bit, result)
    return results

def plot(n_bit, j):
    fig = plt.figure()
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    dot = [':b','-g','--r', '-.c']
    l = 0
    cross_types = [ 'one_point', 'uniform', 'population_wise', 'perfect_mix']
    for i, cross_type in enumerate(cross_types):        
        plt.plot(range(len(exp[i][j])), exp[i][j], dot[i], marker='o', label=cross_type)
        l = max(l, len(exp[i][j]))
    plt.plot(range(l), np.full(l, 0.5762), '#888', label='theoretical')
    plt.yticks(np.arange(0.54, 0.7, step=0.02)) 
    plt.xlabel("generations t")
    plt.ylabel("selction intensity I")
    plt.legend(loc = 'upper right')
    fig.savefig('{}.png'.format(n_bit))
    plt.clf()

def error(exp):
    for j, n_bit in enumerate(n_bits):
        print(n_bit, end=' ')

        for i, cross_type in enumerate(cross_types):
            print(' & ', end='')
            print('{:.4f}'.format(MAPE(exp[i][j])), end='')
        print(" \\\\")

if __name__ == '__main__':
    # ts = time.time()
    # with Pool(len(cross_types)) as p:
    #     exp = list(p.map(job, cross_types))
    #     p.close()
    # te = time.time()
    
    # save(exp, 'intensity.pkl')
    # print('time: {}'.format(te-ts))
    exp = load('intensity.pkl')
    # error(exp)
    for j, n_bit in enumerate(n_bits):
        plot(n_bit, j)