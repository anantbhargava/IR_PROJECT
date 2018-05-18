import os
import argparse
import pickle

import arbit2
import arbitprime
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness - for the purpose of this project use USD-BTC pair for trading (they are also passed as defaut values) - though the program can easily be modified to include all cryto and traditional currencies")

    parser.add_argument("--currency", type=str, help="This the currency you have. For e.g. USD, INR, etc.", default="USD")
    parser.add_argument("--historical", type=int, help="Print the crypto coins available for trade.", default=200)
    parser.add_argument("--trade-currency", type=str, help="Currencies you want to trade in. For e.g. BTC, USD.", default="BTC")
    parser.add_argument("--print-ex", type=str, choices=["T", "F"], help="Do you want to print the exchanges currently being used.", default="F")
    parser.add_argument("--top-ex", type=str, choices=["T", "F"],
                        help="Print the top exchanges by volume currently being used.", default="F")
    parser.add_argument("--coins", type=str, choices=["T", "F"], help="Print the crypto coins available for trade.", default="F")


    args = parser.parse_args()

    return args


def main():
    args = get_args()

    param1 = args.currency
    param2 = args.trade_currency
    param3 = args.historical

    print("calculating historical data:")
    Price, Volume = arbitprime.historical(param2,param1,param3)
    Price = np.asarray(Price)
    Volume = np.asarray(Volume)
    Output = np.vstack((Price, Volume))

    if args.print_ex=="T":
        full = input("do you need exchange data too(y or n)?\n")
        class1 = arbitprime.exchanges(printexchange="T", printdata=full)
        print("PRINTING EXCHANGES:")
        for item in class1:
            print(item)

    if args.top_ex=="T":
        print("PRINTING TOP 10 EXCHANGES BY VOLUME:")
        class2 = arbitprime.topexchange(param1=param1, param2=param2)
        for item in class2:
            print(item)

    if args.coins=="T":
        print("PRINTING TOP 10 EXCHANGES BY VOLUME:")
        arbitprime.coinlist()

    return(Output)

if __name__ == "__main__":
    main()