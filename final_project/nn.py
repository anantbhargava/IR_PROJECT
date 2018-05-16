import keras
from keras.layers import Dense, Conv1D, Dropout, Input, LSTM, Reshape, Flatten, TimeDistributed, MaxPooling1D, Lambda,GlobalMaxPool1D
from keras.layers.merge import Concatenate
from keras import optimizers
from keras.utils import Sequence
from random import shuffle
import glob
from util import load_pickle
import numpy as np
import logging
from keras.callbacks import Callback

def get_test_acc(dataloader, model):
    """
    Get the test accuracy given the dataloader and the model
    """
    curr_loss = np.array([])
    while True:
        in_dat, out_dat = dataloader.loader.__getitem__()
        if dataloader.looping:
            break
        pred_dat = model.predict(in_dat)
        curr_loss = np.concatenate((curr_loss, (pred_dat - out_dat) ** 2))
    logging.info('Test Loss: {}'.format(np.mean(curr_loss)))



class custom_callback(Callback):
    """
    Custom callback to latch on
    """
    def __init__(self, val_thresh,acc_prefix,dataloader):
        super(custom_callback, self).__init__()
        logging.info('Instantiated Callback')
        self.val_thresh = val_thresh
        self.val_fname = acc_prefix + '.txt'
        self.file_prefix = acc_prefix
        self.epoch = -1
        self.loader = dataloader

    def on_epoch_end(self, epoch, logs={}):
        logging.debug('CALLBACK CALLED')
        self.epoch += 1
        self.loader.looping, self.loader.overall_idx = False, 0
        pred_aclabel,pred_inlabel = np.array([]),np.array([])
        while True:
            in_dat, out_dat = self.loader.__getitem__(0)
            if self.loader.looping:
                break

            # Make array of actual and in labels
            pred_inlabel = np.concatenate((pred_inlabel, self.model.predict(in_dat).flatten().astype(np.float32)))
            pred_aclabel = np.concatenate((pred_aclabel,out_dat.flatten().astype(np.float32)))

        # Current loss
        curr_loss = np.mean((pred_aclabel - pred_inlabel)**2)
        with open(self.val_fname, 'a') as fid:
            fid.writelines('{} {} \n'.format(curr_loss, self.epoch))
        with open(self.val_fname, 'a') as fid:
            fid.writelines('{} {} \n'.format(logs.get('loss'), self.epoch))
        if curr_loss > self.val_thresh:
            f_name = self.file_prefix + '_' + str(self.epoch) + '.model'
            self.model.save_weights(f_name)
            logging.info('\n Wrote model val loss: {} , File Name: {} '.format(curr_loss, f_name))
        with open(self.file_prefix + '_' + str(self.epoch) + '.txt','w') as fid:
            for idx in range(pred_inlabel.shape[0]):
                fid.writelines('{} {} \n'.format(pred_inlabel[idx],pred_aclabel[idx]))

class Loader(Sequence):
    def __init__(self, pkl_name, batch_size, max_word, word_dict_pkl, hash_tag_num, handle_num):
        # Load the pickle
        main_dat = load_pickle(pkl_name)
        self.overall_idx = 0
        self.batch_size = batch_size
        self.curr_batch_idx = 0
        self.max_word = max_word
        self.word_dict = load_pickle(word_dict_pkl)
        self.data, self.labels = main_dat['data'], main_dat['labels']
        self.hash_tag_num = hash_tag_num
        self.handle_num = handle_num
        self.looping = False

    def get_data(self, labels, data):
        # Shuffle the data
        main_dat, handle_arr, hashtag_arr = [], [], []
        labels = np.expand_dims(labels, 1)
        max_num_tweet = 503#sum([len(data[idx]) for idx in range(data.shape[0])])
        for idx in range(data.shape[0]):
            # data[idx] is of format [word list], [supp list] = [handle, [hashtags]]
            time_main, time_handle, time_hash_tags = [], [], []
            for segs in data[idx]:
                # Get the word list and the supplementary data list
                word_list, supp_list = segs

                # Get the word vectors expand dim for concatenation
                word_arr = [np.expand_dims(self.word_dict[word], 0) for word in word_list if word in self.word_dict]

                # Trim or expand if necessary
                if len(word_arr) > self.max_word:
                    word_arr = word_arr[:self.max_word]
                else:
                    word_arr += [np.zeros((1, 300))] * (self.max_word - len(word_arr))

                # Make the vector for the tweet
                np_word_arr = np.concatenate(word_arr, axis=0)
                # expand dims for making a batch and add to the main batch
                time_main.append(np.expand_dims(np_word_arr, 0))

                # Get the handle and the hash tag data together
                handle = np.zeros((1, self.handle_num))
                hash_tag = np.zeros((1, self.hash_tag_num))

                # Gets the handle, one hot encoding
                if supp_list[0] is not None:
                    handle[0, supp_list[0]] = 1

                # Add the hash tags
                for ele in supp_list[1]:
                    hash_tag[0, ele] = 1

                # Add to the main array
                time_handle.append(handle)
                time_hash_tags.append(hash_tag)
                # print(handle.shape)

            time_main += [np.zeros((1, self.max_word, 300))] * (max_num_tweet - len(time_main))
            time_handle += [np.zeros((1, self.handle_num))] * (max_num_tweet - len(time_handle))
            time_hash_tags += [np.zeros((1, self.hash_tag_num))] * (max_num_tweet - len(time_hash_tags))

            # Add the data to the main array
            main_dat.append(np.expand_dims(np.concatenate(time_main), axis=0))
            handle_arr.append(np.expand_dims(np.concatenate(time_handle), axis=0))
            hashtag_arr.append(np.expand_dims(np.concatenate(time_hash_tags), axis=0))
        return [np.concatenate(main_dat), np.concatenate(handle_arr), np.concatenate(hashtag_arr)], labels

    def __getitem__(self, item):
        # Adjust the overall index if done with file
        if self.overall_idx > len(self.data):
            random_idx = np.random.choice(len(self.data), len(self.data))
            self.data = self.data[random_idx]
            self.labels = self.labels[random_idx]
            self.overall_idx = 0
            self.looping = True

        self.overall_idx += self.batch_size
        return self.get_data(labels=self.labels[self.overall_idx - self.batch_size: self.overall_idx],
                             data=self.data[self.overall_idx - self.batch_size: self.overall_idx])

    def __len__(self):
        return int(len(self.data) / self.batch_size)


class main_nn:
    """
    nn architecture
    """
    def __init__(self, kernel_dict={3: 80, 4: 60, 5: 60, 6: 60, 8: 60}, num_tweets=10, dropout=0.3, optimizer=None):
        self.shapes = kernel_dict
        self.num_tweets = num_tweets
        self.drop_out = dropout
        self.optim = optimizer

    def produce_model(self):
        input_tweet = Input(shape=(self.num_tweets, 30, 300), name='in_tweet')
        hashtag = Input(shape=(self.num_tweets, 200), name='hashtags')
        handle = Input(shape=(self.num_tweets, 200), name='handles')

        # Convolution Architecture
        k1_raw = TimeDistributed(Conv1D(filters=self.shapes[3], kernel_size=1, padding='same', name='K1'))(input_tweet)
        k2_raw = TimeDistributed(Conv1D(filters=self.shapes[3], kernel_size=2, padding='same', name='K2'))(input_tweet)
        k3_raw = TimeDistributed(Conv1D(filters=self.shapes[3], kernel_size=3, padding='same', name='K3'))(input_tweet)
        k4_raw = TimeDistributed(Conv1D(filters=self.shapes[4], kernel_size=4, padding='same', name='K4'))(input_tweet)
        k5_raw = TimeDistributed(Conv1D(filters=self.shapes[5], kernel_size=5, padding='same', name='K5'))(input_tweet)
        k6_raw = TimeDistributed(Conv1D(filters=self.shapes[6], kernel_size=6, padding='same', name='K6'))(input_tweet)
        k8_raw = TimeDistributed(Conv1D(filters=self.shapes[8], kernel_size=8, padding='same', name='K8'))(input_tweet)
        k1_max = TimeDistributed(GlobalMaxPool1D())(k1_raw)
        k2_max = TimeDistributed(GlobalMaxPool1D())(k2_raw)
        k3_max = TimeDistributed(GlobalMaxPool1D())(k3_raw)
        k4_max = TimeDistributed(GlobalMaxPool1D())(k4_raw)
        k5_max = TimeDistributed(GlobalMaxPool1D())(k5_raw)
        k6_max = TimeDistributed(GlobalMaxPool1D())(k6_raw)
        k8_max = TimeDistributed(GlobalMaxPool1D())(k8_raw)

        # Handle and hashtag processing and reshaping it
        handle_10_vec = TimeDistributed(Dense(units=10, activation='relu'))(handle)
        hashtag_10_vec = TimeDistributed(Dense(units=10, activation='relu'))(hashtag)

        # Output layer
        out_conv = keras.layers.concatenate([k1_max,k2_max,k3_max, k4_max, k5_max, k6_max, k8_max, handle_10_vec, hashtag_10_vec],
                                            name='Combining_Layers')
        int_layer = TimeDistributed(Dense(units=150,activation='relu'))(out_conv)
        dropout_int_layer = TimeDistributed(Dropout(rate=0.4))(int_layer)
        in_lstm = TimeDistributed(Dense(units=100,activation='relu'))(dropout_int_layer)

        # Combined all layers
        # in_lstm_reshaped = Reshape(target_shape=(self.num_tweets, 50),name='Reshape_Conv')(in_lstm)
        out_lstm = LSTM(units=100, name='LSTM')(in_lstm)

        # Linear + Output
        out_linear1 = Dense(units=80, activation='relu', name='Linear_1')(out_lstm)
        out_linear2 = Dense(units=40, name='Linear_2')(out_linear1)
        out = Dense(units=1, name='Final_Layer')(out_linear2)
        model = keras.Model(inputs=[input_tweet, hashtag, handle], outputs=out)
        model.compile(loss='mean_squared_error', optimizer=self.optim)
        return model

def main():
    nn = main_nn(num_tweets=500, optimizer=optimizers.Adam())
    nn.produce_model()
    main_loader = Loader(pkl_name='data/processed_readynn.pkl',batch_size=12,max_word=30,word_dict_pkl='data/wordvectors.pkl',hash_tag_num=200,handle_num=200)
    main_loader.__getitem__(0)

if __name__ == '__main__':
    main()