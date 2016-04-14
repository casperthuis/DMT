import pandas as pd
import seaborn as sns
import numpy as np
from datetime import timedelta

d = pd.read_csv('dataset_mood_smartphone.csv', index_col=2, parse_dates=[2])

# Sample hours
# d0.ix[d0.index.indexer_between_time(datetime.time(9), datetime.time(12))]


dp = d.pivot_table(index=[d.index, d.id], columns='variable', values='value')
dp.reset_index(inplace=True)
dp.index = dp.time
dp = dp.drop('time', 1)

ids = dp.id.unique()

resample_method_dictionary = {
    'activity': np.mean,
    'appCat.builtin': np.sum,
    'appCat.communication': np.sum,
    'appCat.entertainment': np.sum,
    'appCat.finance': np.sum,
    'appCat.game': np.sum,
    'appCat.office': np.sum,
    'appCat.other': np.sum,
    'appCat.social': np.sum,
    'appCat.travel': np.sum,
    'appCat.unknown': np.sum,
    'appCat.utilities': np.sum,
    'appCat.weather': np.sum,
    'call': np.sum,
    'circumplex.arousal': np.mean,
    'circumplex.valence': np.mean,
    'mood': np.mean,
    'screen': np.sum,
    'sms': np.sum,
}

data_per_id = {}
for id in ids:
    data_per_id[id] = dp.loc[dp.id == id].drop('id', 1)
    data_per_id[id] = data_per_id[id].resample('1D').agg(resample_method_dictionary)
    for index in data_per_id[id][data_per_id[id].mood.notnull() == True].index:
        first_day = index - timedelta(days=5)
        last_day = index - timedelta(days=1)
        column = data_per_id[id][first_day:last_day]
        column.index = range(1,6)
        column.pivot(index=column.index,values=column.time_index)
        break
    break
