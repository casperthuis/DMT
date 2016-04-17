from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
import os
import numpy as np
import time
import pickle
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
labels = []
inputs = 0 #will be specified later in the code
hidden = 20
outputs = 1
initialize = True
for i in os.listdir('data'):
    if i.endswith(".csv"): 
        stop = False
        with open('data/' + i) as f:
            totals = np.zeros(len(labels))
            counter = 0
            for counter, line in enumerate(f):
                line = line.replace('\n','')
                line = line.strip()
                line = line.split(',')
                values = line[1:]
                values = line[-20:]
                if len(line) != 97:
                    stop = True
                    break
                if counter == 0:
                    if initialize == True:
                        initialize = False
                        labels = values
                        inputs = len(labels) - outputs
                        ds = SupervisedDataSet(inputs, 1)     
                    totals = np.zeros(len(values))
                    continue
                
                for j, value in enumerate(values):
                    if value == '':
                        values[j] = 0.0
                values = np.array(values, dtype='float32')
                totals = totals + values
                #ds.addSample(values[0:-1], values[-1])
            mean = totals/counter
            if stop == True:
                continue
        with open('data/' + i) as f:
            next(f)
            for counter, line in enumerate(f):
                line = line.replace('\n','')
                line = line.strip()
                line = line.split(',')
                values = line[1:]
                values = line[-20:]
                for j, value in enumerate(values):
                    if value == '':
                        values[j] = 0.0
                values = np.array(values, dtype='float32')
                ds.addSample(values[0:-1], values[-1])
#%%   
  
l = len(ds.getField('input'))    
w =  len(ds.getField('input').transpose())
nonzero = {}
zero = {}
for i in xrange(w):
    nonzero[i] = np.count_nonzero(ds['input'].transpose()[i])
    zero[i] = len(ds.getField('input')) - np.count_nonzero(ds['input'].transpose()[i])
print nonzero
print zero
print labels
d_view = [ (v,k) for k,v in zero.iteritems() ]
d_view.sort(reverse=True) # natively sort tuples by first element
i = 0
for v,k in d_view:
    i += 1
    print 
    print "%s %s: %d zeroes, %d non-zero" % (k,labels[k],v, l-v)