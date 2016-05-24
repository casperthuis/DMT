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
noPatients = 0
standardization = 'yes'
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
                values = list( values[i] for i in [16,35,54,73,92, -1] )
                #this is to remove the mood from the inputs.
#                del values[92]
#                del values[73]
#                del values[54]
#                del values[35]
#                del values[16]

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
            noPatients += 1
        with open('data/' + i) as f:
            next(f)
            for counter, line in enumerate(f):
                line = line.replace('\n','')
                line = line.strip()
                line = line.split(',')
                values = line[1:]
                values = list( values[i] for i in [16,35,54,73,92, -1] )
                #this is to remove the mood from the inputs.
#                del values[92]
#                del values[73]
#                del values[54]
#                del values[35]
#                del values[16]

                #values = line[-20:]
                for j, value in enumerate(values):
                    if value == '':
                        values[j] = 0.0
                values = np.array(values, dtype='float32')
                if standardization == 'yes':
                    ds.addSample(values[0:-1] - mean[0:-1], values[-1] - mean[-1])
                else:
                    ds.addSample(values[0:-1], values[-1])
    break #comment this in to only use the first patient

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
errorVec = []
for i, o in enumerate(out):
    sign = np.sign(out[i] * tstdata['target'][i])
    if sign == -1:
        fout += 1
    else:
        goed += 1
    totaal += 1
    error =  out[i] - tstdata['target'][i]
    errorVec.append(error[0])
    print out[i], tstdata['target'][i], error #, sign
    MAE = MAE + abs(out[i] - tstdata['target'][i])
    MSE = MSE + (out[i] - tstdata['target'][i])**2
MSE = MSE / totaal
MAE = MAE / totaal
#print 'fout', fout
#print 'goed', goed
print 'totaal', totaal
print 'MSE', MSE
print 'MAE', MAE

#%%
import seaborn as sns
print 'Neural network on based on', noPatients, 'patients'
print 'Number of nodes:'
print '   Input:..................', inputs
print '   Hidden:.................', hidden
print '   Output:.................', outputs
print 'Total training instances:..', len(trndata) 
print 'Total test instances:......', len(tstdata)
print 'Mean absolute error:.......', np.mean(np.abs(errorVec))
print 'Mean Squared error:........', MSE[0]
print 'Standardization:...........', standardization
sns.distplot(errorVec, bins=15)
print errorVec
