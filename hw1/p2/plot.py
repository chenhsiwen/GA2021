from scipy.special import comb 
import numpy as np
from util import save, load

max_ell = 1000

def f(s):
    return np.sqrt(2*(np.log(s) - np.log(np.sqrt(4.14*np.log(s)))))
  

def g(s):
   return np.sqrt(2*(np.log(s) - np.log(np.sqrt(4.14*np.log(s)))))
  
if __name__ == '__main__':
    print(f(2))
    print(g(2))


