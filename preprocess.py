import pandas as pd
# import seaborn as sns
import numpy as np
from datetime import timedelta
import pickle

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
    current_d = dp.loc[dp.id == id].drop('id', 1)
    current_d = current_d.resample('1D').agg(resample_method_dictionary)
    data = pd.DataFrame()
    for i in current_d[current_d.mood.notnull() == True].index:
        first_day = i - timedelta(days=5)
        last_day = i - timedelta(days=1)
        column = current_d[first_day:last_day]
        if column.index.size == 5:
            column.index = range(1, 6)
            a = pd.DataFrame(column.stack()).reset_index()
            a.index = a.level_0.map(str) + a.level_1
            a = a.drop(['level_0', 'level_1'], 1).transpose()
        else:
            current_d.drop(i)
        data = data.append(a)
    data['mood_next_day'] = current_d[current_d.mood.notnull() == True].mood.as_matrix()
    data_per_id[id] = data
    data.to_csv('data/data_' + id+'.csv')

with open('data_dict.pickle', 'w') as f:
    pickle.dump(data_per_id, f)