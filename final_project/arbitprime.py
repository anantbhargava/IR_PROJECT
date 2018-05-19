import sys
import os
import requests
import json
import numpy as np

def exchanges(printexchange, printdata):
    g = requests.get("https://min-api.cryptocompare.com/data/all/exchanges")
    data = g.json()
    Exname = []
    if (printexchange == "T"):
        for key in data:
            Exname.append(key)
        if printdata =="y":
            print(data)
            return Exname
        return Exname
    #print(next(iter(data.values())))
    #print(len(data.get("Buda")))

def coinlist():
    g = requests.get("https://www.cryptocompare.com/api/data/coinlist/")
    data = g.json()
    print(data)

def topexchange(param1,param2):
    site = "https://min-api.cryptocompare.com/data/top/exchanges?fsym="+param2+"&tsym="+param1
    g = requests.get(site)
    data = g.json()
    kiss = input("Do you want to print entire data?(y or n)")
    if kiss == "n":
        print(data['Data'])
    Exname = []
    for key in data['Data']:
        Exname.append(key)
        print(key)
    return(Exname)
    #return(data)

def historical(param1,param2,param3):
    site = "https://min-api.cryptocompare.com/data/histohour?fsym="+param1+"&tsym="+param2+"&limit="+str(param3)
    g = requests.get(site)
    data = g.json()
    data1 = data['Data']
    sitevolume = "https://min-api.cryptocompare.com/data/exchange/histohour?tsym="+param1+"&limit="+str(param3)
    vol = requests.get(sitevolume)
    datavolume = vol.json()
    data2 = datavolume['Data']
    price = []
    volume = []
    for i in range(len(data1)-1):
        #print(i,":",data1[i]['close'])
        #price(1,i) = data1[i]['close']
        price.append(data1[i]['close'])
        volume.append(data2[i]['volume'])
    return price, volume
    #for item in data1:
    #    print(data1[item])

def stream():
    g = requests.get("https://min-api.cryptocompare.com/data/subsWatchlist?fsyms=AUD&tsym=BTC")
    data = g.json()
    print(data)

def tradingpairs():
    g = requests.get("https://min-api.cryptocompare.com/data/subs?fsym=BTC&tsyms=INR")
    data = g.json()
    print(data)