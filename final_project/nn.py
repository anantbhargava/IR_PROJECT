import keras
from keras.layers import Dense, Conv1D, Dropout, Input, LSTM, Reshape, Flatten, TimeDistributed, MaxPooling1D, Lambda
from keras.layers.merge import Concatenate
from keras import optimizers
from keras.utils import Sequence
from random import shuffle
import glob
from util import load_pickle
import random
import numpy as np
import logging
from keras.callbacks import Callback
import tensorflow as tf
import keras.backend as K


class Loader(Sequence):
    def __init__(self, pkl_name, batch_size, max_word, word_dict_pkl, hash_tag_num, handle_num):
        # Assumes word dim = 300, max word =
        # Load the pickle
        main_dat = load_pickle(pkl_name)
        self.data, self.labels = main_dat['data'], main_dat['labels']
        self.overall_idx = 0
        self.batch_size = batch_size
        self.curr_batch_idx = 0
        self.max_word = max_word
        self.word_dict = load_pickle(word_dict_pkl)
        self.hash_tag_num = hash_tag_num
        self.handle_num = handle_num

    def get_data(self, labels, data):
        # Shuffle the data
        main_dat, handle_arr, hashtag_arr = [], [], []
        labels = np.expand_dims(labels, 0)

        for idx in range(data):
            # data[idx] is of format [word list], [supp list] = [handle, [hashtags]]
            time_main, time_handle, time_hash_tags = [], [], []
            for segs in data[idx]:
                # Get the word list and the supplementary data list
                word_list, supp_list = segs

                # Get the word vectors expand dim for conctenation
                word_arr = [np.expand_dims(self.word_dict[word], 0) for word in word_list if word in self.word_dict]

                # Trim or expand if necessary
                if len(word_arr) > self.max_word:
                    word_arr = word_arr[:self.max_word]
                else:
                    word_arr += [np.asarray(1, 300)] * (self.max_word - len(word_arr))

                # Make the vector for the tweet
                np_word_arr = np.concatenate(word_arr, axis=0)
                # expand dims for making a batch and add to the main batch
                time_main.append(np.expand_dims(np_word_arr, 0))

                # Get the handle and the hash tag data together
                handle = np.zeros(1, self.handle_num)
                hash_tag = np.zeros(1, self.hash_tag_num)

                # Gets the handle, one hot encoding
                if supp_list[0] is not None:
                    handle[supp_list[0]] = 1

                # Add the hash tags
                for ele in supp_list[1]:
                    hash_tag[ele] = 1

                # Add to the main array
                time_handle.append(handle_arr)
                time_hash_tags.append(hash_tag)

            # Add the data to the main array
            main_dat.append(np.concatenate(time_main))
            handle_arr.append(np.concatenate(time_handle))
            hashtag_arr.append(np.concatenate(time_hash_tags))
        return [np.concatenate(main_dat), np.concatenate(handle_arr), np.concatenate(hashtag_arr)], labels

    def get_all_dat(self):
        return self.get_data(self.labels, self.data)

    def __getitem__(self, item):
        # Adjust the overall index if done with file
        if self.overall_idx > len(self.data):
            random_idx = np.random.choice(len(self.data),len(self.data))
            self.data = self.data[random_idx]
            self.labels = self.labels[random_idx]
            self.overall_idx = 0

        self.overall_idx += self.batch_size
        return self.get_data(labels=self.labels[self.overall_idx - self.batch_size: self.overall_idx],
                                    data=self.data[self.overall_idx-self.batch_size: self.overall_idx])

    def __len__(self):
        return len(self.data)/self.batch_size