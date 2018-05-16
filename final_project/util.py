import pickle
import seaborn as sns
import matplotlib.pyplot as plt



def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, file=f)