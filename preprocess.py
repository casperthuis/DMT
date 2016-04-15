import pandas as pd
# import seaborn as sns
import numpy as np
from datetime import timedelta
import pickle

d = pd.read_csv('dataset_mood_smartphone.csv', index_col=2, parse_dates=[2])

dp = d.pivot_table(index=[d.index, d.id], columns='variable', values='value')
dp.reset_index(inplace=True)
dp.index = dp.time
dp = dp.drop('time', 1)

ids = dp.id.unique()

resample_method_dictionary = {
    'activity': np.mean,
    'appCat.builtin': np.nansum,
    'appCat.communication': np.nansum,
    'appCat.entertainment': np.nansum,
    'appCat.finance': np.nansum,
    'appCat.game': np.nansum,
    'appCat.office': np.nansum,
    'appCat.other': np.nansum,
    'appCat.social': np.nansum,
    'appCat.travel': np.nansum,
    'appCat.unknown': np.nansum,
    'appCat.utilities': np.nansum,
    'appCat.weather': np.nansum,
    'call': np.nansum,
    'circumplex.arousal': np.mean,
    'circumplex.valence': np.mean,
    'mood': np.mean,
    'screen': np.nansum,
    'sms': np.nansum,
}

data_per_id = {}
for id in ids:
    current_d = dp.loc[dp.id == id].drop('id', 1)
    current_d = current_d.resample('1D', ).agg(resample_method_dictionary)
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