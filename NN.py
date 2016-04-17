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
                #values = line[-20:]
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
                #values = line[-20:]
                for j, value in enumerate(values):
                    if value == '':
                        values[j] = 0.0
                values = np.array(values, dtype='float32')
                #ds.addSample(values[0:-1] - mean[0:-1], values[-1] - mean[-1])
                #ds.addSample(values[0:-1]/ mean[0:-1], values[-1]/ mean[-1])
                ds.addSample(values[0:-1], values[-1])

#%%
tstdata, trndata = ds.splitWithProportion( 0.25 )
net = buildNetwork(inputs, hidden, outputs, bias=True) #, hiddenclass=TanhLayer)
trainer = BackpropTrainer(net, trndata, momentum=0.1, weightdecay=0.01) 
start = time.time()
trainer.trainUntilConvergence(maxEpochs=1000, verbose=True, continueEpochs=10, validationProportion=0.25)
print time.time() - start , 'seconds'
#%%
out = net.activateOnDataset(tstdata)
fout = 0
goed = 0
totaal = 0
MSE = 0.0
MAE = 0.0
for i, o in enumerate(out):
    sign = np.sign(out[i] * tstdata['target'][i])
    if sign == -1:
        fout += 1
    else:
        goed += 1
    totaal += 1
    print out[i], tstdata['target'][i], out[i] - tstdata['target'][i]#, sign
    MAE = MAE + abs(out[i] - tstdata['target'][i])
    MSE = MSE + (out[i] - tstdata['target'][i])**2
MSE = MSE / totaal
MAE = MAE / totaal
#print 'fout', fout
#print 'goed', goed
print 'totaal', totaal
print 'MSE', MSE
print 'MAE', MAE

#for i in xrange(1,200):
#    print trainer.train() #UntilConvergence()

#net = buildNetwork(inputs, hidden, outputs)

#print net.activate([1, 0])
#print net.activate([1, 0])
#print net.activate([2, 4])

#%%
#import pickle
#with open('data_dict.pickle', 'r') as f:
#    data_per_id = pickle.load(f)
#%%
#print data_per_id['AS14.01'].columns[-19:-1]
