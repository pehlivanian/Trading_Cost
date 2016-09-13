from __future__ import division
import os
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import nan as NA
import scipy.odr.odrpack as odr

import data as CMEdata

class DataElement(object):
    def __init__(self):
        pass
    def accept(self,visitor):
        visitor.visit(self)

class CMECostModel(DataElement):
    def __init__(self,working_dir="../../data/", zero_intcpt=True, spread_term=False):
        self.working_dir = working_dir
        self.data = CMEdata.CostData()
        if (not zero_intcpt and spread_term):
            print "This particular model is not supported."
            sys.exit()
        if (zero_intcpt):            
            self.model_curve_guess, self.model_curve_ODR, self.num_free_params = \
            CMECostModel.power_law,CMECostModel.power_law_ODR, 2
        elif (spread_term):
            self.model_curve_guess, self.model_curve_ODR, self.num_free_params = \
            CMECostModel.power_law_with_spread,CMECostModel.power_law_with_spread_ODR, 3
        else:
            self.model_curve_guess, self.model_curve_ODR, self.num_free_params = \
            CMECostModel.power_law_const,CMECostModel.power_law_const_ODR, 3
        self.x,self.y = CMECostModel.getmodel_variables(self)

    def setTradedata(self,tradedata):
        self.data.setTradedata(tradedata)
        self.x,self.y = CMECostModel.getmodel_variables(self)        

    @staticmethod
    def power_law_ODR(B,x):
        return B[0] * x**B[1]

    @staticmethod
    def power_law_with_spread_ODR(B,x):
        return B[0] * x[0]**B[1] + x[1]*B[2]

    @staticmethod
    def power_law_with_spread(x,a,b,c):
        return a*x[0]**b + c*x[1]

    @staticmethod
    def power_law(x,a,b):
        return a * x**b

    @staticmethod
    def power_law_const_ODR(B,x):
        return B[0] + B[1] * x**B[2]

    @staticmethod
    def power_law_const(x,a,b,c):
        return a + b * x**c
    
    def getmodel_variables(self):
        trades = self.data.tradedata
        y = 10000 * trades['sidesign'] * (trades['avgprc'] - trades['ap'])/trades['ap']
        x = trades['durpov'] * np.sqrt(trades['variance'] * trades['durfrac'])        
        return x,y

    def qc(self,num_cuts,truncate=False,taillimit=.95):
        x,y = self.x, self.y
        
        if (truncate):
            df = pd.DataFrame({'x' : x, 'y' : y})
            df = df.sort_values(by=['x'])
            x = df['x'][0:int(taillimit*len(df))]
            y = df['y'][0:int(taillimit*len(df))]
        
        m = []
        s = []
        xaxis = []

        # pandas qcut ver 0.17 doesn't seem to be
        # backward-compatible
        qct0 = pd.qcut(x,num_cuts,labels=False)
        qct1 = pd.qcut(x,num_cuts,retbins=True)
        cnt = pd.DataFrame({ 'label' : qct0}).groupby('label').count()
        labels = np.sort(np.unique(qct0))
        for label in labels:
            m.append(np.mean(y[qct0 == label]))
            s.append(np.sqrt(np.var(y[qct0 == label])))
        m = [0] + m
        s = [0] + s
        xaxis.append(0)
        for i in [xx+1 for xx in range(len(qct1[1]))[:-1]]:
            xaxis.append(.5*(qct1[1][i-1] + qct1[1][i]))
        # To throw out the last bin
        # if (truncate):
        #     x = x[qct0 != labels[-1]]
        #     y = y[qct0 != labels[-1]]
        #     xaxis = xaxis[:-1]
        #     m = m[:-1]
        #     s = s[:-1]

        return x,y,xaxis,m,s,cnt
    
    def fit(self,num_cuts,truncate=True,withSpreadterm=False):
        x,y,xaxis,m,s,cnt = CMECostModel.qc(self,num_cuts,truncate)
        try:
            popt,pcov = curve_fit(self.model_curve_guess,x,y)
        except RuntimeError:
            popt = [1.0] * CMECostModel.num_free_params
        func = odr.Model(self.model_curve_ODR)
        odrdata = odr.Data(x,y)
        odrmodel = odr.ODR(odrdata,func,beta0=popt,maxit=500,ifixx=[0])
        o = odrmodel.run()
        return o,x,y,xaxis,m,s,cnt
    
    def predict(self,o,x):
        return self.model_curve_ODR(o.beta,x)
        
