import numpy as np 
from multiprocessing import Pool
import time
from util import save, load

def int2bit(num, n_bits):
     bit = np.full(n_bits, 0)
     for i in range(n_bits):
          bit[i] = num % 2
          num = num // 2
     return np.array(bit)

def bit2int(arr):
     return arr.dot(2**np.arange(arr.size))

def initFitness(n_bits):
     return np.random.rand(np.power(2,n_bits)).reshape(np.full(n_bits, 2))

def switch(fitness, n_bits):
     opt = int2bit(np.argmax(fitness), n_bits)
     for i, o in enumerate(opt[::-1]):
          if not o:
               if i == 0:
                    fitness[[0,1]] = fitness[[1,0]]
               if i == 1:
                    fitness[:,[0,1]] = fitness[:,[1,0]]
               if i == 2:
                    fitness[:,:,[0,1]] = fitness[:,:,[1,0]] 
               if i == 3:
                    fitness[:,:,:,[0,1]] = fitness[:,:,:,[1,0]] 
     return fitness
     
def testDeception(fitness, n_bits):
     fitness = switch(fitness, n_bits)
     opt = int2bit(np.argmax(fitness), n_bits)
     trap = 1 - opt
     for i in range(fitness.size - 2):
          schema = int2bit(i+1, n_bits)
          axis = tuple(np.argwhere(schema).flatten())
          _axis = tuple(np.argwhere(1-schema).flatten())
          idx = 0
          for j, a in enumerate(_axis):
               idx += trap[a] * 2**j
          mean = np.mean(fitness, axis=axis)
          if (idx != np.argmax(mean)):  
               return 0
     return 1 

result = []
num_exp = 20
n_bits = 4
samples = int(1e6)

def job(r): 
    counter = 0
    for t in range(samples):
        fitness = initFitness(n_bits)
        if(testDeception(fitness, n_bits)):
            counter += 1
        if t %100000 == 0: 
            print(r, t, counter/(t+1))
    return counter/samples

if __name__ == '__main__':
    ts = time.time() 
    with Pool(num_exp) as p:
        data = list(p.map(job, [i for i in range(num_exp)]))
    p.close()
    te = time.time()   
    tc = te - ts 
    print('mean:{0:.8f}, std:{1:.8f}'.format(np.mean(data), np.std(data)))
    print('time cost: {}s'.format(tc))
