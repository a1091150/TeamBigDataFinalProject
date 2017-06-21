# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

import tensorflow as tf
color = sns.color_palette()

#another machine learning tools
from sklearn import preprocessing, model_selection

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)
print '===============================Start'
train_df = pd.read_csv("./data/train.csv")

#print train_df['life_sq'].values
#print train_df.'shape
#print train_df.head()

plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('price', fontsize=12)
#plt.show()
plt.savefig("outputfig/price_count.png")

plt.figure(figsize=(12,8))
sns.distplot(train_df.price_doc.values, bins=50, kde=True)
plt.xlabel('price', fontsize=12)
#plt.show()
plt.savefig("outputfig/price_count.png")


#計算Csv中NaN的個數

missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
#plt.show()
plt.savefig("outputfig/missing_count_data.png")

#Error Field, try to use tf.one_hot(...) but failed
#將非數字的欄位 轉換成 編號形式，不過錯誤。
'''
depth = tf.constant(train_df.shape[1])#292
for i in train_df.columns:
    if train_df[i].dtype =='object':
        print train_df[i].as_matrix()
        inputdata = tf.constant(train_df[i].as_matrix())
        one_hot_encoded = tf.one_hot(indices = inputdata , depth = depth)
        with tf.Session() as sess:
            print (sess.run(one_hot_encoded))
            print (one_hot_encoded.eval())
'''

#顯示資料表的重要性

for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))

train_y = train_df.price_doc.values
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
#plt.show()
plt.savefig("outputfig/importantance.png")

# 每年月的房價中位數
plt.figure(figsize=(12,8))
sns.barplot(grouped_df.yearmonth.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Year Month', fontsize=12)
plt.xticks(rotation='vertical')
#plt.show()
plt.savefig("outputfig/yearmonthmedian.png")
print '===============================End'
