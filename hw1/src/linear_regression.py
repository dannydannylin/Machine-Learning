import numpy as np
from sklearn import model_selection, linear_model
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/machine.data")
df.replace('?', np.nan, inplace = True)  # 把 "?" 變成 Nan
df.dropna(inplace = True) # 把缺失值拿掉
df.drop(['vendor_name'], 1, inplace = True) # 把位在"行"的 vendor_name 拿掉
df.drop(['Model_Name'], 1, inplace = True) # 把位在"行"的 Model_Name 拿掉

X = np.array(df.drop(['PRP'], 1)) # 把位在"行"的 PRP 拿掉
Y = np.array(df['PRP']) # 把 PRP 抽出來

# X 為要劃分的特徵集 Y 為要劃分的樣本結果

X_train,X_test,Y_train,Y_test = \
model_selection.train_test_split(X, Y, test_size = 0.3)

accuracynum = 0
for j in range(1,11):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)

    accuracynum = accuracynum + regr.score(X_test, Y_test)
accuracynum = accuracynum / 10

print( accuracynum )



