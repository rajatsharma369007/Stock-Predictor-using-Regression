import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(dates,prices)
plt.scatter(dates,prices,color='red')
plt.plot(dates,rfr.predict(dates),color='blue')
plt.show()
print(rfr.predict(31))
print(answer)
