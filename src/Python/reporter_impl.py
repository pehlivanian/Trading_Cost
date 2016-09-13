from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
from matplotlib import mlab, cm
from scipy import stats
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression

class reporter_impl(object):
    def __init__(self):
        pass

    @staticmethod
    def overlap_test(trades):
        itime,tottime = 0,0
        t = trades.sort_values(by='tstart')
        start_times,end_times = t['tstart'],t['tend']
        intervals = zip(start_times,end_times)
        for interval in intervals:
                start_time,end_time = interval
                tottime += (end_time - start_time).seconds
                int_start_times = start_times[(start_times < end_time) & (start_times > start_time)]
                if (len(int_start_times) > 0):
                        int_start_times = int_start_times.reset_index(drop=True)
                        int_start_time = int_start_times[0]
                        itime += (end_time - int_start_time).seconds
                        
        return 1.0-(itime/tottime)

    @staticmethod
    def direction_test(trades):
        preds = (trades['sidesign'] * (trades['endprctxmax'] - trades['ap']) > 0)
        return sum(preds>0) / len(preds)

    @staticmethod
    def convexity_test(trades):
                preds = (trades['sidesign'] * (trades['avgprc']
                                                                           - .5*(trades['endprctxmax'] + trades['ap'])) >= 0)
                return sum(preds>0)/len(preds)

    @staticmethod
    def describe_test(trades):
        x,y = model_variables(trades)
        trades['X'] = x
        trades['y'] = y
        return trades[['y','X']].describe()

    @staticmethod
    def OLS_test(trades):
        x,y = model_variables(trades)
        results = sm.OLS(y,x).fit()
        return results.params[0]

    @staticmethod
    def plot_count(r, titleStr, pdf):
        plt.title(titleStr)
        pdf.savefig()
        plt.close()
        r.plot(kind='bar',color='y',legend=True)
        plt.title('Pct Non-Overlapping Trades by Class')
        pdf.savefig()
        plt.close()

    @staticmethod
    def plot_cuts(ax, x, num_bins, row_num, col_num, nameStr):
        try:
            q = pd.qcut(x,num_bins,retbins=True)
            ax[row_num,col_num].plot(q[1][1:],'o-')
            ax[row_num,col_num].set_xlabel(nameStr + ': Cut Number')
            ax[row_num,col_num].set_ylabel('Value at Cut')
            ax[row_num,col_num].set_title(nameStr + ': Percentiles')
        except ValueError:
            ax[row_num,col_num].text(0.2,0.8, 'Insufficient Variation in Data', rotation=45)
            ax[row_num,col_num].set_xlabel(nameStr + ': Cut Number')
            ax[row_num,col_num].set_ylabel('Value at Cut')
            ax[row_num,col_num].set_title(nameStr + ': Percentiles')

    @staticmethod
    def overlap_count(trades):
        r = list()
        trades['trade'] = 1
        grouped = trades.groupby('sym')
        tmp = grouped.agg(reporter_impl.overlap_test)['trade']
        df1 = pd.DataFrame({ 'sym' : tmp.index, 'pct' : 100*tmp.values})
        df2 = trades[['inst','class','sym']]
        df = pd.merge(df1,df2,on='sym')
        r.append(df.groupby('inst').mean())
        r.append(df.groupby('class').mean())
        return r

    @staticmethod
    def direction_count(trades):
        r = list()
        trades['trade'] = 1
        grouped = trades.groupby(['sym'])
        r.append(grouped.agg(reporter_impl.direction_test)['trade'])
        grouped = trades.groupby(['inst'])
        r.append(grouped.agg(reporter_impl.direction_test)['trade'])
        grouped = trades.groupby(['class'])
        r.append(grouped.agg(reporter_impl.direction_test)['trade'])
        grouped = trades.groupby(['targetstrat'])
        r.append(grouped.agg(reporter_impl.direction_test)['trade'])
        return r

    @staticmethod
    def convexity_count(trades):
        r = list()
        trades['trade'] = 1
        grouped = trades.groupby(['sym'])
        r.append(grouped.agg(reporter_impl.convexity_test)['trade'])
        grouped = trades.groupby(['inst'])
        r.append(grouped.agg(reporter_impl.convexity_test)['trade'])
        grouped = trades.groupby(['class'])
        r.append(grouped.agg(reporter_impl.convexity_test)['trade'])
        grouped = trades.groupby(['targetstrat'])
        r.append(grouped.agg(reporter_impl.convexity_test)['trade'])
        return r

    @staticmethod
    def QualityofFittables(DataElement,pdf,outpath,truncate,nameStr=''):
        groupCats = ['class','targetstrat']
        tradedata = DataElement.data.tradedata.copy()
        mod = DataElement

        for groupCat in groupCats:
        
                r = pd.DataFrame()
                grouped = tradedata.groupby(groupCat)
                for name,data in grouped:
                        mod.setTradedata(data)
                        nameStrLoc = name
                        if (len(nameStr) > 0):
                                nameStrLoc = name + "_" + nameStr
                        r = pd.concat([r,
                                                   reporter_impl.QualityofFitgrouptables(
                                                           mod,
                                                           '/home/charles/',
                                                           truncate=truncate,
                                                           nameStr=nameStrLoc)])
                try:
                        r = np.round(r[['Group','const (est)','beta (est)',
                                                        'gamma (est)','lin r^2','pow_law r^2']],4)
                except KeyError:
                        r = np.round(r[['Group','beta (est)',
                                                        'gamma (est)','lin r^2','pow_law r^2']],4)


                from pandas.tools.plotting import table
                fig,ax = plt.subplots(1,1)
                table(ax,r,loc='upper right',colWidths=[0.1,0.15,0.15,0.15,0.15,0.15])
                df = pd.DataFrame(r['gamma (est)'])
                df.index = r['Group']
                df.plot(kind='bar',color='y',ax=ax,ylim=(0,2),legend=False)
                plt.ylabel('Gamma (est)')
                plt.title('Coefficient by ' + groupCat)
                pdf.savefig()
                plt.close()

        # DataElement.setTradedata(tradedata)

    @staticmethod
    def QualityofFitgrouptables(DataElement,outpath,truncate,nameStr=''):
        r = pd.DataFrame()
        
        o,x,y,xaxis,m,s,cnt = DataElement.fit(10,truncate=truncate)
        # Check OLS fit
        # scipy linregress insufficient
        # s,intercept,r_val,p_val,std_err = stats.linregress(y,DataElement.predict(o,x))

        # Redo this section
        lin_fit = sm.OLS(y,x).fit()
        lin_pred = sm.OLS(y,lin_fit.predict(x.reshape(-1,1))).fit()
        r_2_lin = lin_pred.rsquared
        pl_fit = sm.OLS(y,DataElement.predict(o,x)).fit()
        r_2 = pl_fit.rsquared

        if (len(o.beta) > 2):
                r = pd.concat([r, pd.DataFrame({'Group'                        : [nameStr],
                                                                                'truncated'                : [truncate],
                                                                                'count'                        : [len(x)],
                                                                                'const (est)'          : [o.beta[0]],
                                                                                'beta (est)'           : [o.beta[1]],
                                                                                'gamma (est)'          : [o.beta[2]],
                                                                                'const (std err)'  : [o.sd_beta[0]],
                                                                                'beta (std err)'   : [o.sd_beta[1]],
                                                                                'gamma (std err)'  : [o.sd_beta[2]],
                                                                                'pow_law r^2'          : [r_2],
                                                                                'lin r^2'                  : [r_2_lin],
                                                                                'res_var'                  : [o.res_var],
                                                                                'sum_sq error'         : [o.sum_square],
                                                                                'sum_sq eps'           : [o.sum_square_eps] })])
        else:
                r = pd.concat([r, pd.DataFrame({'Group'                        : [nameStr],
                                                                                'truncated'                : [truncate],
                                                                                'count'                        : [len(x)],
                                                                                'beta (est)'           : [o.beta[0]],
                                                                                'gamma (est)'          : [o.beta[1]],
                                                                                'beta (std err)'   : [o.sd_beta[0]],
                                                                                'gamma (std err)'  : [o.sd_beta[1]],
                                                                                'pow_law r^2'          : [r_2],
                                                                                'lin r^2'                  : [r_2_lin],
                                                                                'res_var'                  : [o.res_var],
                                                                                'sum_sq error'         : [o.sum_square],

                                                                                'sum_sq eps'           : [o.sum_square_eps] })])
                
        r.to_csv(outpath + 'model_fitstab_' + nameStr + '.csv',sep=',')
        
        return r

    @staticmethod
    def QualityofFitplots(DataElement,pdf,outpath,truncate,nameStr=''):

        ftrades = DataElement.data.tradedata

        o,x,y,xaxis,m,s,cnt = DataElement.fit(25,truncate=truncate)
        reporter_impl.sc_plot(x,y,nameStr)
        pdf.savefig()
        plt.close()

        reporter_impl.xdescribe(DataElement,nameStr,pdf)

        num_cuts = [10*x for x in range(1,13)]
        num_cuts[10] = 125
        num_cuts[11] = 150
        for i in range(len(num_cuts)):
                n_c = num_cuts[i]
                o,x,y,xaxis,mn,sd,cnt = DataElement.fit(n_c,truncate=truncate)

                if (np.mod(i,4) == 0):
                        fig,ax = plt.subplots(nrows=2, ncols=2)
                        row = 0
                        col = 0
                else:
                        row = 1*(np.mod(i,4) >= 2)
                        col = (np.mod(i,2))
                ax[row,col].errorbar(xaxis,mn,yerr=0,fmt='r*-',label='Empirical Cost')
                ax[row,col].plot(xaxis,DataElement.model_curve_ODR(o.beta,xaxis),'k--',label='Model Cost')
                if (row == 0) and (col == 0):
                        ax[row,col].legend()
                        ax[row,col].set_title(nameStr + ":: (Beta,Gamma) = (" +
                                                                  str(np.round(o.beta[0],4)) + ", "
                                                                  + str(np.round(o.beta[1],4)) + ")")
                else:
                        ax[row,col].set_title("(num_cuts) = (" + str(n_c) + ")")
                        
                if (row == 1):
                        ax[row,col].set_xlabel('Feature')
                if (col == 0):
                        ax[row,col].set_ylabel('Cost in Bps')
                if (row == 1) and (col == 1):
                   pdf.savefig()
                   plt.close()

    @staticmethod
    def sc_plot(x,y,nameStr=''):
        nullfmt = NullFormatter()                 # no labels

        x = list(x)
        y = list(y)

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
                
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        
        axScatter.scatter(x, y)
        axScatter.set_title('Scatter y on x')

        # Limits
        xmax = np.max(np.fabs(x))
        xbinwidth = xmax / 20
        xlim = (int(xmax/xbinwidth) + 1) * xbinwidth
        xbins = np.arange(-xlim, xlim + xbinwidth, xbinwidth)
        ymax = np.max(np.fabs(y))
        ybinwidth = ymax / 20
        ylim = (int(ymax/ybinwidth) + 1) * ybinwidth
        ybins = np.arange(-ylim, ylim + ybinwidth, ybinwidth)
        
        axHistx.hist(x, bins=xbins)
        axHisty.hist(y, bins=ybins, orientation='horizontal')
        
        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())

        axHistx.set_title(nameStr + ':: x')
        axHisty.set_title(nameStr + '::y')

        # plt.show()

    @staticmethod
    def xdescribe(DataElement,nameStr,pdf):
        num_bins = 20

        fig,ax = plt.subplots(nrows=2, ncols=2)

        x = DataElement.x
        nameStr = nameStr + '::Feature Space'
        reporter_impl.plot_cuts(ax, x, num_bins, 0, 0, nameStr)
        
        x = DataElement.y
        nameStr = nameStr + '::Cost'
        reporter_impl.plot_cuts(ax, x, num_bins, 0, 1, nameStr)

        
        x = DataElement.data.tradedata['dur']
        nameStr = nameStr + '::TradeDuration'
        reporter_impl.plot_cuts(ax, x, num_bins, 1, 0, nameStr)
                
        x = DataElement.data.tradedata['pov']
        nameStr = nameStr + '::Pct of Volume'
        reporter_impl.plot_cuts(ax, x, num_bins, 1, 1, nameStr)

        pdf.savefig()
        plt.close()
        fit,ax = plt.subplots(nrows=2, ncols=2)

        x = DataElement.data.tradedata['size']
        nameStr = nameStr + '::Size in Lots'
        reporter_impl.plot_cuts(ax, x, num_bins, 0, 0, nameStr)
        
        x = DataElement.data.tradedata['durfrac']
        nameStr = nameStr + '::Duration as Pct of Day'
        reporter_impl.plot_cuts(ax, x, num_bins, 0, 1, nameStr)

        pdf.savefig()
        plt.close()

