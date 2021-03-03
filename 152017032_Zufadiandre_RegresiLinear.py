#libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # Import train_test_split function
import numpy as np

X_train = np.array([28,20,21,23,25,19,17,15,22,19,18])
y_train = np.array([21,24,27,22,24,25,28,30,25,27,26])
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# create linear regression object
model_reg = LinearRegression()
# train the model using the training sets
model_reg.fit(X_train, y_train)

# regression coefficients
print('Nama : Zulfadiandre')
print('NRP : 152017032')
print('Coefficients b = {}'.format(model_reg.coef_))
print('Constant a ={} '.format(model_reg.intercept_))
print("Kinerja Pegawai jika tingkat stres 10 = {}".format(model_reg.predict([[10]])))