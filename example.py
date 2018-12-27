import pandas as pd
from sklearn.metrics import mean_squared_error

from tinygbt import GBT

print('Load data...')
df_train = pd.read_csv('./data/regression.train', header=None, sep='\t')
df_test = pd.read_csv('./data/regression.test', header=None, sep='\t')

y_train = df_train[0].values
y_test = df_test[0].values
X_train = df_train.drop(0, axis=1).values
X_test = df_test.drop(0, axis=1).values

print('Start training...')
gbt = GBT(n_estimators=20)
gbt.fit(X_train, y_train, valid_set=(X_test, y_test), early_stopping_rounds=5)

print('Start predicting...')
y_pred = gbt.predict(X_test)

print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
