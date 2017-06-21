import pandas as pd
import numpy as np


#train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
for i in test.columns.values:
    nandata = test[test[i].isnull()].index
    test.loc[nandata,i] = 0

test.to_csv('./outputcsv/test_nonull.csv',index=False)
