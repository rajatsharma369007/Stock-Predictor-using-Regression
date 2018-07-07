import csv
import numpy as np
from sklearn.tree import DecisionTreeRegressor
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
decision_t = DecisionTreeRegressor(random_state=0)
decision_t.fit(dates,prices)
plt.scatter(dates,prices,color='red')
plt.plot(dates,decision_t.predict(dates),color='blue')
plt.show()
prediction = decision_t.predict(31)
print(prediction)
print(answer)

