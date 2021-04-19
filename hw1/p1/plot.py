from scipy.special import comb 
import numpy as np
from util import save, load

max_ell = 1000

def f(p, k):
    n =  np.log(0.5) / np.log(1-p)
    for ell in range(4, max_ell):
        c = comb(ell, k)
        if (c > n):
            print(k, ell)
            break   

p_3 = 5.749*1e-3
p_4 = 2.985*1e-4
if __name__ == '__main__':
    data = []
    f(p_3, 3)
    f(p_4, 4)
    c = np.log(1-p_3)/np.log(1 -p_4)
    print(4*c +3)
    for ell in range(4, 10000000):
        if (c*4/(ell-3) < 1 ):
            print(ell)
            break


