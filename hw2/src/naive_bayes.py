import numpy as np
from sklearn import model_selection, naive_bayes
import pandas as pd
import matplotlib.pyplot as plt

# question3 (1)
df = pd.read_csv("../data/breast-cancer-wisconsin.data")
df.replace('?', np.nan, inplace = True)  # 把 "?" 變成 Nan
df.dropna(inplace = True)  # 把缺失值拿掉
df.drop(['id'], 1, inplace = True)  # 把位在"行"的 id 拿掉

X = np.array(df.drop(['Class'], 1))  # 把位在"行"的 class 拿掉
Y = np.array(df['Class'])  # 把 class 抽出來

# X 為要劃分的特徵集 Y 為要劃分的樣本結果

X_train, X_test, Y_train, Y_test = \
model_selection.train_test_split(X, Y, test_size = 0.3)

k = []
accuracy = []
accuracynum = 0

for j in range(1, 11):
    clf = naive_bayes.GaussianNB()
    clf.fit( X_train, Y_train )
    accuracynum = clf.score(X_test, Y_test)
    k.append( j )
    accuracy.append(accuracynum)

plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(k, accuracy)
plt.show()

