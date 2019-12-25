import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(r'D:\Machine-Learning\Test.csv')
dataset.info()

x = dataset.iloc[:, 1: 4]
y = dataset['Sales']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size = 0.8)

#create and fit the model for prediction
lin = LinearRegression()
lin.fit(X_train, Y_train)
y_pred = lin.predict(X_test)

#create coefficients
coef = lin.coef_
components = pd.DataFrame(list(zip(x.columns, coef)), columns = ['component', 'value'])
components = components.append({'component': 'intercept', 'value': lin.intercept_}, ignore_index = True)
