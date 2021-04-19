
import pickle
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

def save(data, name = 'hw1.pkl'):
    filename = os.path.join(dir_path, 'exp', name)
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)

def load(name = 'hw1.pkl'):
    filename = os.path.join(dir_path, 'exp', name)
    with open(filename, 'rb') as fp:
        return pickle.load(fp)