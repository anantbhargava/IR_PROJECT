import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import keras
import nn


def main_run(args):
    model_class = nn.main_nn(num_tweets=args.num_tweets,optimizer=keras.optimizers.Adam(lr=args.lr))
    model = model_class.produce_model()
    print(model.summary())
    logging.info('Instantiated Model')

    # Get the weights for the model
    if args.load is not None:
        model = keras.models.load_model(args.load)
        logging.info('Loaded weights from file: {}'.format(args.load))
    if args.test:
        # Loading the dataloader
        loader = nn.Loader('data/test.pkl', batch_size=6, max_word=args.max_w, word_dict_pkl=args.word_dict,
                           hash_tag_num=args.hash_tag, handle_num=args.handle_num)
        nn.get_test_acc(loader, model)
        exit()

    # Instantiating the callback and tensorboard which are then put in tensorboard
    val_loader = nn.Loader('data/val.pkl', batch_size=36, max_word=args.max_w, word_dict_pkl=args.word_dict,hash_tag_num=args.hash_tag, handle_num=args.handle_num)
    tr_loader = nn.Loader('data/train.pkl', batch_size=36, max_word=args.max_w, word_dict_pkl=args.word_dict,hash_tag_num=args.hash_tag, handle_num=args.handle_num)
    logging.info('Instantiated Data Loaders')
    callback = nn.custom_callback(val_thresh=args.val_thresh, dataloader=val_loader, acc_prefix=args.acc_prefix)
    logging.info('Callback instantiated, data generator instantiated')
    tb = keras.callbacks.TensorBoard(histogram_freq=0, batch_size=args.bs)
    model.fit_generator(generator=tr_loader, epochs=args.ep, callbacks=[callback, tb])


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Prediction Arr')
    parser.add_argument("-load", type=str, default=None, help='File to load weights')
    parser.add_argument("-val_thresh", type=float, default=0.5, help='Threshold after which weights should be saved')
    parser.add_argument("-lr", type=float, default=0.001, help='Learning Rate For Adam Optimizer')
    parser.add_argument("-ep", type=int, default=10, help='Number of epochs')
    parser.add_argument("-max_w", type=int, default=30, help='Maximum words')
    parser.add_argument("-num_tweets", type=int, default=300, help='Number of tweets allowed in the network')
    parser.add_argument("-bs", type=int, default=256, help='Batch size')
    parser.add_argument("-log", type=str, default="INFO", help='Log level')
    parser.add_argument("-acc_prefix", type=str, default='lol', help='Acc prefix')
    parser.add_argument("-hash_tag", type=int, default=200, help='Number of hash tags being monitored')
    parser.add_argument("-handle_num", type=int, default=200, help='Number of handles being monitored')

    parser.add_argument("-test", type=bool, default=False, help='Test Mode')
    parser.add_argument("-word_dict", type=str, default="data/wordvectors.pkl", help='Word Vector Pickle')

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', level=args.log)
    # Main Function
    main_run(args)
