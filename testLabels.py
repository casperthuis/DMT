# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:36:19 2016

@author: Wolf
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
import os
import numpy as np
import time
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
labels = []
inputs = 0
hidden = 50
outputs = 1
initialize = True
for i in os.listdir('data'):
    if i.endswith(".csv"): 
        with open('data/' + i) as f:
            totals = np.zeros(len(labels))
            counter = 0
            for counter, line in enumerate(f):
                line = line.replace('\n','')
                line = line.strip()
                line = line.split(',')
                values = line[1:]
                if len(values) == 96:
                    print '--->',i, len(values)
                break
                    
