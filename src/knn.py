import numpy as np
from sklearn import model_selection, neighbors
import pandas as pd
import matplotlib.pyplot as plt

# question3
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
for i in range(3, 16):
    k.append(i)
    for j in range(1,11):
        clf = neighbors.KNeighborsClassifier(i)
        clf.fit( X_train, Y_train )

        accuracynum = accuracynum + clf.score(X_test, Y_test)
    accuracynum = accuracynum / 10
    accuracy.append(accuracynum)
    accuracynum = 0

plt.plot(k, accuracy)
plt.show()


# question4

# df = pd.read_csv("../data/breast-cancer-wisconsin.data")
# df.replace('?', np.int64( 0 ), inplace = True)  # 把 "?" 變成 0
# df.dropna(inplace = True)  # 把缺失值拿掉
#
# dic = {}
#
#
#
# for i in range( 1, df.shape[1] - 1, 1 ):
#     for j in range( i + 1, df.shape[1] - 1, 1 ) :
#         attribute = df.columns[i] + " & " + df.columns[j]
#         cov =  pd.to_numeric( df[ df.columns[i] ] ).corr( pd.to_numeric( df[ df.columns[j] ] ) )
#         dic.update( { attribute : cov } )
#
# so = sorted( dic.items(), key = lambda d:d[1], reverse = True )
#
#
# for item in so:
#     print( item )


