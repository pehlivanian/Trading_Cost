from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
from matplotlib import mlab, cm
from scipy import stats
from sklearn.linear_model import LinearRegression
from reporter_impl import *

class DataElementVisitor(object):
    def __init__(self):
        self.impl = reporter_impl()
    def visit(self,DataElement):
        pass

class BoltvStrobeReporter(DataElementVisitor):
    def __init__(self,outpath,nameStr='',pdf=None,closeonExit=True):
        super(BoltvStrobeReporter, self).__init__()
        self.outpath = outpath
        self.nameStr = nameStr
        self.pdf = pdf
        self.closeonExit = closeonExit
    def visit(self,DataElement):
        BoltvStrobereport(self,DataElement)
        if (self.closeonExit):
            self.pdf.close()

class PrelimStatsReporter(DataElementVisitor):
    def __init__(self,outpath,nameStr='',pdf=None,closeonExit=True):
        super(PrelimStatsReporter, self).__init__()
        self.outpath = outpath
        self.nameStr = nameStr
        self.pdf = pdf
        self.closeonExit = closeonExit
    def visit(self,DataElement):
        PrelimStatsreport(self,DataElement)
        if (self.closeonExit):
            self.pdf.close()

class QualityofFitReporter(DataElementVisitor):
    def __init__(self,outpath,nameStr='',qc=True,pdf=None,closeonExit=True):
        super(QualityofFitReporter, self).__init__()
        self.outpath = outpath
        self.nameStr = nameStr
        self.qc = qc
        self.pdf = pdf
        self.closeonExit = closeonExit
    def visit(self,DataElement):
        QualityofFitreport(self,DataElement)
        if (self.closeonExit):
            self.pdf.close()
            
class Reporter(DataElementVisitor):    
    def __init__(self,outpath='.',nameStr='',qc=True,pdf=None,closeonExit=True):
        self.outpath = outpath
        self.nameStr = nameStr
        self.qc = qc
        self.pdf = pdf
        self.closeonExit = closeonExit
        self.reporters = [ BoltvStrobeReporter(outpath,nameStr,pdf,False),
                           PrelimStatsReporter(outpath,nameStr,pdf,False),
                           QualityofFitReporter(outpath,nameStr,qc,pdf,False)]

    def visit(self,DataElement):
        for reporter in self.reporters:
            reporter.visit(DataElement)
        if (self.closeonExit):
            self.pdf.close()

def BoltvStrobereport(self,DataElement):
    ftrades = DataElement.data.tradedata

    if (self.pdf == None):
        pdf = PdfPages(self.outpath + "TargetstratCount.pdf")
    else:
        pdf = self.pdf
    
    df = pd.DataFrame(ftrades.groupby(['inst']).count()['volume'])
    df.columns = ['Trade count']
    df.plot(kind='bar',color='y',legend=True)
    plt.title('Count by Instrument (CME Futures Symbol)')
    pdf.savefig()
    plt.close()

    df = pd.DataFrame(ftrades.groupby(['class']).count()['volume'])
    df.columns = ['Trade count']
    df.plot(kind='bar',color='y',legend=True)
    plt.title('Count by Class {AG, EN, EQ, FX, IR, MT}')
    pdf.savefig()
    plt.close()

    df = pd.DataFrame(ftrades.groupby(['targetstrat']).count()['volume'])
    df.columns = ['Trade count']
    df.plot(kind='bar',color='y',legend=True)
    plt.title('Count : All Trades {BOLT, STROBE}')
    pdf.savefig()
    plt.close()


def PrelimStatsreport(self,DataElement):

    ftrades = DataElement.data.tradedata

    if (self.pdf == None):
        pdf = PdfPages(self.outpath + "ParentOrderProfile.pdf")
    else:
        pdf = self.pdf

    r1 = self.impl.overlap_count(ftrades)
    r2 = self.impl.direction_count(ftrades)
    r3 = self.impl.convexity_count(ftrades)

    self.impl.plot_count(r1[0], 'Pct Non-Overlapping Trades by Inst (CME Symbol)', pdf)
    self.impl.plot_count(r2[1], 'Pct Direction Match by Inst', pdf)
    self.impl.plot_count(r2[3], 'Pct Direction Match by Targetstrat', pdf)
    self.impl.plot_count(r3[1], 'Pct Convexity Match by Inst', pdf)
    self.impl.plot_count(r3[2],'Pct Convexity Match by Class', pdf)
    self.impl.plot_count(r3[3], 'Pct Convexity Match by Targetstrat', pdf)

def QualityofFitreport(self,DataElement):
    if (self.pdf == None):
        pdf = PdfPages(self.outpath + 'ModelfitDiagnostics.pdf')
    else:
        pdf = self.pdf
        
    self.impl.QualityofFitplots(DataElement,pdf,self.outpath,self.qc,self.nameStr)
    self.impl.QualityofFittables(DataElement,pdf,self.outpath,self.qc,self.nameStr)
    grouped = DataElement.data.tradedata.groupby('class')
    for name,data in grouped:
        DataElement.setTradedata(data)
        self.impl.QualityofFitplots(DataElement,pdf,self.outpath,self.qc,nameStr=name)
