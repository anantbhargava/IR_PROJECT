from util import load_pickle
from util import save_pickle
import numpy as np
import spacy
import pandas as pd
import ujson as json
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from collections import Counter
import multiprocessing as mp


def get_prices(f_name):
    with open(f_name, 'r') as fid:
        indv_points = fid.read().split('\n')
    prev_time, store_data = 0, []
    for ele in indv_points:
        if len(ele) != 0:
            main_dict = json.decode(ele)

            # Check that the data is in ascending order of time
            assert (main_dict['timestamp'] > prev_time)
            prev_time = main_dict['timestamp']
            # Store the data
            store_data.append({'change': main_dict['ticker']['change'], 'time': main_dict['timestamp'],
                               'price': main_dict['ticker']['price']})
    logging.info('Completed getting prices from: {}, entries: {}'.format(f_name, len(store_data)))
    return store_data


def convert_timestamp(input_str):
    """
    Convert to time unix format
    """
    try:
        out_arr = input_str.split('-')
        int_arr = out_arr[2].split('T')
        time_split = int_arr[1].split(':')
        fin_f = list(map(int, out_arr[:-1] + [int_arr[0]] + time_split))
        return int(datetime.datetime(fin_f[0], fin_f[1], fin_f[2], fin_f[3], fin_f[4], fin_f[5]).strftime("%s"))
    except (ValueError, IndexError):
        return None


def main_pkl(f_name, out_fname):
    # Read in the file
    fid = open(f_name, 'r')
    out_arr = fid.read().split('\n')
    process_data = []

    # Loop over all the data
    for ele in out_arr:
        twit_split = ele.split('||')

        # Check if the data has the correct format (3 ||)
        if len(twit_split) != 4:
            logging.info('Twitter sample: {}'.format(ele))
            continue
        assert (len(twit_split[-1]) == 0)
        # Convert timestamp and add to process_data
        time_stamp = convert_timestamp(twit_split[-2])
        if time_stamp:
            process_data.append({'handle': twit_split[0], 'text': twit_split[1], 'time': time_stamp})
        else:
            logging.debug('Time Stamp Not Detected: {}'.format(ele))

    save_pickle({'dat': process_data}, out_fname)
    logging.info('Length of raw data: {} process data: {} pickle name:{}'.format(
        len(out_arr), len(process_data), out_fname))


def handle_analyzer(f_name, img_name, out_fname):
    # Counting tweets
    tweet_arr, handle_imp = load_pickle(f_name)['dat'], Counter()
    logging.info('Going through tweets now')
    for tweet in tweet_arr:
        handle_imp[tweet['handle']] += 1
    plot_save_dat(handle_imp, out_fname, img_name, 'Number of tweets', 'Probablity')
    logging.info('Saved histogram with number of tweets from handle vs. freq to: {}'.format(img_name))


def plot_save_dat(counter, out_fname, img_name, xlabel, ylabel):
    with open(out_fname, 'w') as fid:
        for ele in counter.most_common():
            fid.writelines('%s  %d\n' % (ele[0], ele[1]))
        logging.info('Wrote number of tweets from handle to file: {}'.format(out_fname))
    plt.clf()
    # Histogram plot
    plt.hist(np.array(list(counter.values())), bins=100, normed=True)
    plt.xlabel(xlabel)
    plt.yscale('log')
    plt.ylabel(ylabel)
    plt.savefig(img_name)


def hashtag_analyzer(f_name, img_name, out_fname):
    # Counting tweets
    tweet_arr, hashtag_imp = load_pickle(f_name)['dat'], Counter()
    logging.info('Going through tweets now')
    for tweet in tweet_arr:
        main_text = tweet['text'].split()
        for word in main_text:
            if word[0] == '#':
                hashtag_imp[word] += 1
    plot_save_dat(hashtag_imp, out_fname, img_name, 'Number of reptitions', 'Probablity')
    logging.info('Saved histogram with occurance of hashtag vs. freq to: {}'.format(img_name))


def clean_tweet(tweet):
    """
    Return hashtags and other words separately
    """
    word_out, hashtags = [], []
    for word in tweet.split():
        if word[0] == '#':
            hashtags.append(word)
        elif ((len(word) != 0) and (word[0] != '@')) and (
                len(word) < 4 or ((len(word) > - 4) and (word[:4] != 'http'))):
            word_out.append(word)
    return word_out, hashtags


def make_dict_pickle(f_name, out_fname):
    logging.info('Making pickle for the dictionary')
    word_set = set()
    for tweet in load_pickle(f_name)['dat']:
        words, _ = clean_tweet(tweet['text'])
        for word in words:
            word_set.add(word)
    logging.info('Number unique words: {}'.format(len(word_set)))
    save_pickle(word_set, out_fname)
    logging.info('Saved dictionary to: {}'.format(out_fname))


def parallel_word_dict(w_list, st, end):
    logging.info('Parallel process start')
    w_list = w_list[st:end]
    nlp, out_dict, count = spacy.load('en_core_web_lg'), {}, 0
    print('loaded')
    for word in w_list:
        word_obj = nlp(word)
        if word_obj.has_vector:
            out_dict[word] = word_obj.vector
        count += 1
        if count % 500 == 0:
            print(count)
    return out_dict


def make_wordvec_dict(f_name, out_fname, threads):
    # Make list of unique words
    word_list = list(load_pickle(f_name))

    # Send job to workers
    per_f = int(len(word_list) / threads) + 1
    logging.info('Per Thread {}'.format(per_f))
    pool = mp.Pool(processes=threads)
    processes = [
        pool.apply_async(parallel_word_dict, args=(word_list, per_f * (x - 1), per_f * x)) for x in
        range(1, threads + 1)]

    # Get data and put it out
    output = [process.get() for process in processes]
    out_dict = {}
    for ele in output:
        out_dict = {**out_dict, **ele}
    pool.close()
    save_pickle(out_dict, out_fname)
    logging.info('Made Dictionary Using Spacy')


def make_main_process_pkl(prices_fname, word_pkl, hashtag_fname, handle_fname, out_fname):
    def get_dict(fname):
        out_set, tot_count = {}, 0
        with open(fname, 'r') as fid:
            word_arr = fid.read().split('\n')
            for ele in word_arr:
                if len(ele) > 0:
                    out_set[(ele.split('\t')[0])] = tot_count
                    tot_count += 1
        return out_set

    # Get prices
    prices_dict = get_prices(f_name=prices_fname)
    # Get the dictionaries and the sets
    main_arr, hashtag_dict, handle_dict = load_pickle(word_pkl)['dat'], get_dict(hashtag_fname), get_dict(handle_fname)
    # Sort the stuff
    sorted(main_arr, key=lambda val: val['time'])
    # Main Storage, and index for time array
    dat_arr, lab_arr, time_idx, samples,time_arr = [], [], 0, [],[]

    # Current slot storage
    curr_dat, curr_lab = [], None
    num = 0
    for ele in main_arr:
        num += 1
        try:
            # To test when jump more than 1 timestep
            loop_more_than_once = False
            # If current time is higher then jump to next entry, update the arrays
            while ele['time'] >= prices_dict[time_idx]['time']:
                if loop_more_than_once:
                    logging.warning('Jumping one extra timestep for: {}'.format(ele['time']))
                else:
                    time_arr.append(prices_dict[time_idx]['time'])
                    lab_arr.append(curr_lab)
                    dat_arr.append(curr_dat)
                curr_dat, curr_lab = [], None
                time_idx += 1
                loop_more_than_once = True
        except IndexError:
            logging.warning('Ran out of the prices.txt file at tweet index: {}, time index: {}'.format(num, time_idx))
            break

        # Assert that the time stamp on tweet before the current time
        assert(ele['time'] < prices_dict[time_idx]['time'])

        words, hashtag_arr = clean_tweet(tweet=ele['text'])
        hashtag_arr = [hashtag_dict[hashtag] for hashtag in hashtag_arr if hashtag in hashtag_dict]
        handle_num = None
        if ele['handle'] in handle_dict:
            handle_num = handle_dict[ele['handle']]
        curr_dat.append((words, [handle_num, hashtag_arr]))
        curr_lab = prices_dict[time_idx]['change']
    # Ensure that the length of the data and the number of labels are same
    assert(len(dat_arr) == len(lab_arr) == len(time_arr))
    logging.info('Total Samples: {}'.format(len(dat_arr)))

    # Get stats regarding number of tweets per time step and timestep data
    timestep_out = np.asarray([time_arr[idx] - time_arr[idx - 1] for idx in range(1,len(time_arr))])
    number_tweets = np.asarray([time_arr[idx] - time_arr[idx - 1] for idx in range(1,len(time_arr))])

    plt.hist(timestep_out, bins=1000, normed=True)
    plt.xlabel('Time Step')
    plt.yscale('Probablity')
    plt.ylabel(ylabel)
    plt.savefig(img_name)





logging.basicConfig(level='INFO')
# get_prices("data/prices.txt")
# main_pkl("data/tweets_raw.txt", "data/process_dat.pkl")
# handle_analyzer("data/process_dat.pkl", "data/handle_stats.png", 'data/handle_stats.txt')
# hashtag_analyzer("data/process_dat.pkl", "data/hashtag_stats.png", 'data/hashtag_stats.txt')
# make_dict_pickle("data/process_dat.pkl", "data/word_dict.pkl")
# make_wordvec_dict("data/word_dict.pkl", "data/wordvectors.pkl", 20)
make_main_process_pkl(prices_fname="data/prices.txt", word_pkl="data/process_dat.pkl",
                      hashtag_fname="data/hashtag_200.txt", handle_fname="data/handle_stats.txt",out_fname="processed_readynn.pkl")