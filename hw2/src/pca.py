import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# question 5

df = pd.read_csv("../data/exam.txt")

# 把 type 轉成 float ( astype(float) )
X = df.drop(['Class'], 1)  # 把位在"行"的 class 拿掉
Y = df['Class'] # 把 class 抽出來
# 正規劃
Standard_X = StandardScaler().fit_transform(X)

pca = PCA( n_components = 2 )

principalComponents = pca.fit_transform(Standard_X)

variance = pca.explained_variance_ratio_

var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)

plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(60,100.5)
plt.style.context('seaborn-whitegrid')

print( variance )

plt.plot(var)
# plt.show()


# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2'])
#
# finalDf = pd.concat([principalDf, Y], axis = 1)

# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = [2.0, 4.0]
# colors = ['r', 'g']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['Class'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()
# plt.show()