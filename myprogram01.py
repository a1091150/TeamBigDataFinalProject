import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('./data/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('./data/test.csv',parse_dates=['timestamp'])
bad_index = train[train.life_sq>train.full_sq].index
train.loc[bad_index,"life_sq"]=np.NaN

equal_index=[601,1896,2791]
bad_index=train[train.life_sq<5].index
train.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.full_sq < 5].index
train.loc[bad_index, "full_sq"] = np.NaN

kitch_is_build_year = [13117]
train.loc[kitch_is_build_year, "build_year"] = train.loc[kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq > 300].index
train.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
train.product_type.value_counts(normalize= True)
bad_index = train[train.build_year < 1500].index
train.loc[bad_index, "build_year"] = np.NaN
bad_index = train[train.num_room == 0].index
train.loc[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train.loc[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
train.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.floor == 0].index
train.loc[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = train[train.floor > train.max_floor].index
train.loc[bad_index, "max_floor"] = np.NaN

train.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train.loc[bad_index, "floor"] = np.NaN
train.material.value_counts()
train.state.value_counts()
bad_index = train[train.state == 33].index
train.loc[bad_index, "state"] = np.NaN

# brings error down a lot by removing extreme price per sqm
train.loc[train.full_sq == 0, 'full_sq'] = 50
train = train[train.price_doc/train.full_sq <= 600000]
train = train[train.price_doc/train.full_sq >= 10000]

# Add month-year
month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
train['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
train['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
train['month'] = train.timestamp.dt.month
train['dow'] = train.timestamp.dt.dayofweek


# Other feature engineering
train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)

train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
train['room_size'] = train['life_sq'] / train['num_room'].astype(float)


rate_2016_q2 = 1
rate_2016_q1 = rate_2016_q2 / .99903
rate_2015_q4 = rate_2016_q1 / .9831
rate_2015_q3 = rate_2015_q4 / .9834
rate_2015_q2 = rate_2015_q3 / .9815
rate_2015_q1 = rate_2015_q2 / .9932
rate_2014_q4 = rate_2015_q1 / 1.0112
rate_2014_q3 = rate_2014_q4 / 1.0169
rate_2014_q2 = rate_2014_q3 / 1.0086
rate_2014_q1 = rate_2014_q2 / 1.0126
rate_2013_q4 = rate_2014_q1 / 0.9902
rate_2013_q3 = rate_2013_q4 / 1.0041
rate_2013_q2 = rate_2013_q3 / 1.0044
rate_2013_q1 = rate_2013_q2 / 1.0104
rate_2012_q4 = rate_2013_q1 / 0.9832
rate_2012_q3 = rate_2012_q4 / 1.0277
rate_2012_q2 = rate_2012_q3 / 1.0279
rate_2012_q1 = rate_2012_q2 / 1.0279
rate_2011_q4 = rate_2012_q1 / 1.076
rate_2011_q3 = rate_2011_q4 / 1.0236
rate_2011_q2 = rate_2011_q3 / 1
rate_2011_q1 = rate_2011_q2 / 1.011

#train 2015

train_2015_q4_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
# train.loc[train_2015_q4_index, 'price_doc'] = train.loc[train_2015_q4_index, 'price_doc'] * rate_2015_q4
train.loc[train_2015_q4_index, 'average_q_price'] = rate_2015_q4


train_2015_q3_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
#train.loc[train_2015_q3_index, 'price_doc'] = train.loc[train_2015_q3_index, 'price_doc'] * rate_2015_q3
train.loc[train_2015_q3_index, 'average_q_price'] = rate_2015_q3


train_2015_q2_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
#train.loc[train_2015_q2_index, 'price_doc'] = train.loc[train_2015_q2_index, 'price_doc'] * rate_2015_q2
train.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2


train_2015_q1_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
#train.loc[train_2015_q1_index, 'price_doc'] = train.loc[train_2015_q1_index, 'price_doc'] * rate_2015_q1
train.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1


# train 2014
train_2014_q4_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
#train.loc[train_2014_q4_index, 'price_doc'] = train.loc[train_2014_q4_index, 'price_doc'] * rate_2014_q4
train.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4

train_2014_q3_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
#train.loc[train_2014_q3_index, 'price_doc'] = train.loc[train_2014_q3_index, 'price_doc'] * rate_2014_q3
train.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3

train_2014_q2_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
#train.loc[train_2014_q2_index, 'price_doc'] = train.loc[train_2014_q2_index, 'price_doc'] * rate_2014_q2
train.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2

train_2014_q1_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
#train.loc[train_2014_q1_index, 'price_doc'] = train.loc[train_2014_q1_index, 'price_doc'] * rate_2014_q1
train.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1



# train 2013
train_2013_q4_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
# train.loc[train_2013_q4_index, 'price_doc'] = train.loc[train_2013_q4_index, 'price_doc'] * rate_2013_q4
train.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4

train_2013_q3_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
# train.loc[train_2013_q3_index, 'price_doc'] = train.loc[train_2013_q3_index, 'price_doc'] * rate_2013_q3
train.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3

train_2013_q2_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
# train.loc[train_2013_q2_index, 'price_doc'] = train.loc[train_2013_q2_index, 'price_doc'] * rate_2013_q2
train.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2

train_2013_q1_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
# train.loc[train_2013_q1_index, 'price_doc'] = train.loc[train_2013_q1_index, 'price_doc'] * rate_2013_q1
train.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1



# train 2012
train_2012_q4_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
# train.loc[train_2012_q4_index, 'price_doc'] = train.loc[train_2012_q4_index, 'price_doc'] * rate_2012_q4
train.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4

train_2012_q3_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
# train.loc[train_2012_q3_index, 'price_doc'] = train.loc[train_2012_q3_index, 'price_doc'] * rate_2012_q3
train.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3

train_2012_q2_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
# train.loc[train_2012_q2_index, 'price_doc'] = train.loc[train_2012_q2_index, 'price_doc'] * rate_2012_q2
train.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2

train_2012_q1_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
# train.loc[train_2012_q1_index, 'price_doc'] = train.loc[train_2012_q1_index, 'price_doc'] * rate_2012_q1
train.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1


# train 2011
train_2011_q4_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
# train.loc[train_2011_q4_index, 'price_doc'] = train.loc[train_2011_q4_index, 'price_doc'] * rate_2011_q4
train.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4

train_2011_q3_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
# train.loc[train_2011_q3_index, 'price_doc'] = train.loc[train_2011_q3_index, 'price_doc'] * rate_2011_q3
train.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3

train_2011_q2_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
# train.loc[train_2011_q2_index, 'price_doc'] = train.loc[train_2011_q2_index, 'price_doc'] * rate_2011_q2
train.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2

train_2011_q1_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
# train.loc[train_2011_q1_index, 'price_doc'] = train.loc[train_2011_q1_index, 'price_doc'] * rate_2011_q1
train.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1

train['price_doc'] = train['price_doc'] * train['average_q_price']
# train.drop('average_q_price', axis=1, inplace=True)
