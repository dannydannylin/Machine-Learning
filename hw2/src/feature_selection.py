import pandas as pd
import numpy as np
from sklearn import neighbors
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split

# question 4

df = pd.read_csv("../data/breast-cancer-wisconsin.data")
df.replace('?', np.nan, inplace = True)  # 把 "?" 變成 Nan
df.dropna(inplace = True)  # 把缺失值拿掉
df.drop(['id'], 1, inplace = True)  # 把位在"行"的 id 拿掉

X = df.drop(['Class'], 1)  # 把位在"行"的 class 拿掉
Y = df['Class'] # 把 class 抽出來

# X 為要劃分的特徵集 Y 為要劃分的樣本結果

# 50% 的 train, 30% 的 test, 50% 的 find
X_train, X_test_find, Y_train, Y_test_find = \
train_test_split(X, Y, test_size = 0.5)

X_find, X_test, Y_find, Y_test = \
train_test_split(X_test_find, Y_test_find, test_size = 0.6)

accuracynum = 0

for j in range(1, 11):
    # 使用 knn(3) 篩選出前 3 個 feature

    feature_selector = SequentialFeatureSelector(neighbors.KNeighborsClassifier(3),
               k_features = 3,
               forward = True,
               verbose = 2,
               scoring = 'roc_auc',
               cv = 4)

    # 找出是哪 3 個 feature

    features = feature_selector.fit(X_find, Y_find)

    filtered_features= X_find.columns[list(features.k_feature_idx_)]

    # 用那些 feature 做訓練, 算出分數

    clf = neighbors.KNeighborsClassifier(3)
    clf.fit( X_train[filtered_features], Y_train )

    accuracynum = accuracynum + clf.score(X_test[filtered_features], Y_test)

accuracynum = accuracynum / 10

print()
print( accuracynum )

