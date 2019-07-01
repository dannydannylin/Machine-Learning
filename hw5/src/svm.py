import pandas as pd
from sklearn import svm

df = pd.read_csv("../data/input.txt")

X = df.drop(['Class'], 1).astype(float)  # 把位在"行"的 class 拿掉
Y = df['Class'].astype(float) # 把 class 抽出來

svc = svm.SVC(kernel='linear')
svc.fit(X, Y)

print( svc.coef_ )

print( svc.support_vectors_ )
