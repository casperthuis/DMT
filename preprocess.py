import pandas as pd
import seaborn as sns
import numpy as np
import datetime

d = pd.read_csv('dataset_mood_smartphone.csv', index_col=2, parse_dates=[2])

# Sample only mood from one patient
d0 = d.loc[d.id=='AS14.01'].loc[d.variable=='mood']
# Set time as its index
d0.index = d0.time
del d0['time']
# d0.resample('D')

# Sample hours
d0.ix[d0.index.indexer_between_time(datetime.time(9), datetime.time(12))]


dp = d.pivot_table(index=d.index,columns='variable',values='value')
