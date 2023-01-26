import pandas as pd
pd.set_option('display.max_colwidth', None)
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

columns_name=['user_id','item_id','rating','timestamp']

# /home/hwiric/Internship/LightGCN_MovieLens/u.data
df = pd.read_csv("/home/hwiric/Internship/LightGCN_MovieLens/u.data",sep="\t",names=columns_name)

# print(len(df))

# print(df.info())

# 셀에서 동작하는 명령어인 것 같음
# display(df.head(5))

df = df[df['rating']>=3]

# print("Rating Distribution")
df.groupby(['rating'])['rating'].count()

train, test = train_test_split(df.values, test_size=0.2, random_state = 16)
train = pd.DataFrame(train, columns = df.columns)
test = pd.DataFrame(test, columns = df.columns)
# print("Train Size  : ", len(train))
# print("Test Size : ", len (test))

# Train Size  :  66016
# Test Size :  16504


# Label Encoding the User and Item IDs
le_user = pp.LabelEncoder()
le_item = pp.LabelEncoder()

train['user_id_idx'] = le_user.fit_transform(train['user_id'].values)
train['item_id_idx'] = le_item.fit_transform(train['item_id'].values)

train_user_ids = train['user_id'].unique()
train_item_ids = train['item_id'].unique()

# print(len(train_user_ids), len(train_item_ids))

test = test[(test['user_id'].isin(train_user_ids)) & (test['item_id'].isin(train_item_ids))]
# print(len(test))

# 943 1546
# 16472


test['user_id_idx'] = le_user.transform(test['user_id'].values)
test['item_id_idx'] = le_item.transform(test['item_id'].values)

n_users = train['user_id_idx'].nunique()
n_items = train['item_id_idx'].nunique()
# print("Number of Unique Users : ", n_users)
# print("Number of unique Items : ", n_items)

# Number of Unique Users :  943
# Number of unique Items :  1546

# ! <class 'pandas.core.series.Series'>
# print(type(train['user_id_idx']))
# print(type(test['user_id_idx']))

# print(train['user_id_idx'])
# print(test['user_id_idx'])

# ! <class 'pandas.core.frame.DataFrame'>
'''
user_id  item_id  rating  timestamp  user_id_idx  item_id_idx
0          770      250       5  875971902          769          249
1          169      331       5  891268491          168          329
2          327      143       4  888251408          326          142
3           85     1101       4  879454046           84         1086
4          548      264       4  891043547          547          263
# ...        ...      ...     ...        ...          ...          ...
66011      807      177       4  892705191          806          176
66012      145       12       5  882182917          144           11
66013      602      748       3  888638160          601          740
66014      622     1078       3  882671160          621         1063
66015       60       47       4  883326399           59           46
'''
# print(type(train))

# ? print(train['user_id'][770])      →   134 

interected_items_df = train.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()

def sample_neg(x):
    while True:
        neg_id = random.randint(0, 1546 - 1)
        if neg_id not in x:
            return neg_id

indices = [x for x in range(943)]

if 943 < 1024:
    users = [random.choice(indices) for _ in range(1024)]
else:
    users = random.sample(indices, 1024)

users.sort()

users_df = pd.DataFrame(users,columns = ['users'])

interected_items_df = pd.merge(interected_items_df, users_df, how = 'right', left_on = 'user_id_idx', right_on = 'users')

pos_items = interected_items_df['item_id_idx'].apply(lambda x : random.choice(x)).values

neg_items = interected_items_df['item_id_idx'].apply(lambda x: sample_neg(x)).values

print(list(users))

