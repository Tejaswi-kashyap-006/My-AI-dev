from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from doctest import script_from_examples
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

# import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# converting the data dependable variable to 2D array
y = y.reshape(len(y), 1)

# Feature scalling
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# training the svr model on whole dataset
regressor = SVR(kernel='rbf')
regressor.fit(x, y)


# predicting a new result
arr = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))

print(arr)


# visualising the svr result
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(
    regressor.predict(x)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Higher resolution and smoother curve
X_grid = np.arange(min(sc_x.inverse_transform(x)),
                   max(sc_x.inverse_transform(x)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(
    regressor.predict(sc_x.transform(X_grid))), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
