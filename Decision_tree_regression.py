from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, -1].values


# training the decision tree regression model
reggressor = DecisionTreeRegressor(random_state=0)
reggressor.fit(x, y)

# predicting a new result
arr = reggressor.predict([[6.5]])

print(arr)

# visualing the results
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(X_grid, reggressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
