# import requests

# if __name__ == '__main__':
#   api_url = "https://api.opensea.io/api/v1/collection/sandbox"
#   response = requests.get(api_url)
#   print(response.json())
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np


# https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/

if __name__ == '__main__':

  # Price fluctuations over 4 time periods with gas price
  data = {'30-day':  [20, 25, 20, 19],
          '7-day': [5, 10, 7, 2],
          '1-day': [2, 20, -10, -2],
          'gas':[20, 30,60, 6],
          }

  df = pd.DataFrame(data)

  print (df)

  # prices
  X = df.iloc[:, 0:4].values
  
  # gas price
  y = df.iloc[:, 2].values
  print(y)


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  print(X_train)
  print(X_test)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  regressor = RandomForestRegressor(n_estimators=200, random_state=0)
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

  # X, y = make_classification(n_samples=1000, n_features=4,
  #                           n_informative=2, n_redundant=0,
  #                           random_state=0, shuffle=False)
  # clf = RandomForestClassifier(max_depth=2, random_state=0)
  # clf.fit(X, y)
  # print(clf.predict([[0, 0, 0, 0]]))