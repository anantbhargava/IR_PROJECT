import csv
import json
import pandas as pd
import pickle
import numpy as np
import spacy
import logging


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, file=f)