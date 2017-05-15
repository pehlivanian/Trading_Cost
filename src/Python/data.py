from __future__ import division
import string
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import nan as NA
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.parser import parse
import scipy.odr.odrpack as odr
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager

import pdb

working_dir = "../../data/"

class Singleton(object):
    def __new__(self):
        if not hasattr(self, "instance"):
            self.instance = super(Singleton,self).__new__(self)
        return self.instance

class CostData(Singleton):
    def __init__(self):
        self.tradedata = trades(self)
        filter_trades(self)
    def setTradedata(self,tradedata):
        self.tradedata = tradedata
    def trades(self):
        pass
    def filter_trades(self):
        pass

def trades(self):
    fn = working_dir + "t.csv"

    secdef = __secdef()
    vpred = __vpred()
    
    trades  = pd.read_csv(fn, header=0,
                          parse_dates=[0,6,7,15,16],
                          date_parser=my_ts_parser, sep=",")
    
    # Process NaNs
    trades = trades.dropna(how='any',subset=trades.columns.drop('limpx'))
    secdef = secdef.dropna(how='any')
    vpred =  vpred.dropna(how='any')

    # Merge in additional sec def data
    trades = pd.merge(trades,secdef[['inst','class',
                                     'dispfactor','minpxincr',
                                     'lotsize','crossrate',
                                     'dolconv','hrsinday']],
                      on='inst')

    # Add sidesign
    trades['sidesign'] = np.where(trades['side'] == 'BUY',1,-1)

    # Add volume, variance
    trades = pd.merge(trades,vpred[['date','inst','volume','variance']],
                      on=['date','inst'])
    # Adjust variance
    trades['dolsperTick'] = trades['lotsize'] * trades['minpxincr'] * trades['dolconv'] * trades['dispfactor']
    trades['variance'] = 1e-4 * trades['variance'] * pow(trades['dolsperTick'],2) 

    # Modify all prices to be in $
    price_fields = ['avgprc', 'ap', 'endprc', 'endprctxmax', 'twap',
                    'twaptxmax', 'vwap', 'vwaptxmax', 'sweeppx']    
    for price_field in price_fields:
        trades[price_field] *= trades['dispfactor'] * trades['dolconv']

    # Add $ notional field
    trades['dolnot'] = trades['size'] * trades['lotsize'] * trades['ap']

    # Add pov, duration pov
    trades['dur'] = pd.Series([(x-y).seconds for x,y in zip(trades['tend'],trades['tstart'])])
    trades['durfrac'] = pd.Series(trades['dur']/ (60 * 60 * trades['hrsinday']))
    trades['pov'] = 100 * trades['size'] / (trades['volume'] * trades['durfrac'])
    trades['durpov'] = trades['pov'] * trades['durfrac']

    return trades


def my_ts_parser(col):
    r = []
    for c in col:
        try:
            r.append(parse(c))
        except:
            r.append(NA)
    return r

def __lotsize():
    fn = working_dir + "iii.csv"

    r = pd.read_csv(fn, header=0,sep=",")
    return r

def dolconv_():
    fn = working_dir + "dc.csv"

    r = pd.read_csv(fn, header=0, sep=",")
    return r

def __secdef():
    fn = working_dir + "ii.csv"
    
    r = pd.read_csv(fn, header=0, sep=",")

    # Join in conversion factor, lotsizes
    secdef_add = __lotsize()
    r = pd.merge(r,secdef_add[['inst','lotsize','crossrate',
                               'dolconv','hrsinday']],on='inst')
    
    return r

def __vpred():
    fn = working_dir + "v.csv"

    r = pd.read_csv(fn, header=0,
                    parse_dates = [0],
                    date_parser=my_ts_parser, sep=",")

    return r


def filter_trades(self):
    # Basic filter
    # Only use BOLT trades
    self.tradedata = self.tradedata[self.tradedata['targetstrat'] == 'BOLT']
    # $ notional > $1e5
    # self.tradedata = self.tradedata[self.tradedata['dolnot'] > 1e5]
    # Size <= 10% of predicted daily volume
    # self.tradedata = self.tradedata[self.tradedata['size'] < .10 * self.tradedata['volume']]
    # No limit price trades
    self.tradedata = self.tradedata[pd.isnull(self.tradedata['limpx'])]
    # No trades > 1 trading day in duration
    # self.tradedata = self.tradedata[self.tradedata['durfrac'] <= 1]

