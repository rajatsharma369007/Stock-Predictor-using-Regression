import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(row[0][8:10])
			prices.append(float(row[1]))
	return
get_data('daily_GOOGL.csv') # calling get_data method by passing the csv file to it
dates = np.arange(30,0,-1)
dates = np.reshape(dates,(len(dates),1))# converting to matrix of n X 1
answer = prices[29]
prices = prices[30:60]
svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
svr_lin = SVR(kernel= 'linear', C= 1e3)
svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
svr_rbf.fit(dates, prices) # fitting the data points in the models
svr_lin.fit(dates, prices)
svr_poly.fit(dates, prices)
plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
rbf_p,linear_p,poly_p = svr_rbf.predict(31)[0], svr_lin.predict(31)[0], svr_poly.predict(31)[0]
print(str(rbf_p) + ',' + str(linear_p) + ',' + ',' + str(poly_p))
print(answer)
