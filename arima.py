from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
import os
import numpy as np
import time
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from statsmodels.tsa.arima_model import *
labels = []
moods = []
mMSE = 0
totalPatients = 0
for i in os.listdir('data'):
    if i.endswith(".csv"): 
        with open('data/' + i) as f:
            next(f)
            for counter, line in enumerate(f):
                line = line.replace('\n','')
                line = line.strip()
                line = line.split(',')
                mood = line[-1]
                moods.append(mood)

        endog = np.array(moods, dtype='float32')
        #mean = np.mean(endog)
        #endog = endog

        p = 1
        d = 0
        q = 0
        arima = ARIMA(endog, order=(p,d,q))
        model = arima.fit()
        result = model.predict()
        
        endog = endog[d:]
        
        fout = 0
        goed = 0
        totaal = 0
        MSE = 0.0
        MAE = 0.0
        for i, o in enumerate(result):
            sign = np.sign(result[i] * endog[i])
            if sign == -1:
                fout += 1
            else:
                goed += 1
            totaal += 1
            #print out[i], endog[i], result[i] - endog[i]#, sign
            MAE = MAE + abs(result[i] - endog[i])
            MSE = MSE + (result[i] - endog[i])**2
        MSE = MSE / totaal
        MAE = MAE / totaal
        mMSE = mMSE + MSE
        totalPatients += 1
        #print 'fout', fout
        #print 'goed', goed
        #print 'totaal', totaal
        print 'MSE', MSE
        #print 'MAE', MAE
        
        #prediction = results.predict(start=1,end=len(x)-1,exog=x.drop(endogenous,axis=1))
        
        #def objfunc(order, exog, endog):
        #    from statsmodels.tsa.arima_model import ARIMA
        #    fit = ARIMA(endog, order, exog).fit()
        #    return fit.aic()
        #
        #from scipy.optimize import brute
        #grid = (slice(1, 3, 1), slice(1, 3, 1), slice(1, 3, 1))
        #brute(objfunc, grid, args=(exog, endog), finish=None)
print 'average MSE:' , mMSE / totalPatients
