#mostly drafts

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from permetrics.regression import RegressionMetric
import os

#data prep --------------------------------------------------------------------------------------------
first_timestamp = '2024-04-01 00:00:00'
last_timestamp = '2024-04-28 23:00:00'

b21 = pd.read_csv('files/B21_04_2024.csv')
embassy_sensor = pd.read_csv('files/Astana_PM2.5_2024_04.csv')

time_difference = pd.Timedelta(hours=6) #for some reason time offset at airnow is 7 hours
b21['Timestamp'] = pd.to_datetime(b21['Timestamp']) + time_difference
embassy_sensor['Date (LT)'] = pd.to_datetime(embassy_sensor['Date (LT)']) 

b21 = b21.set_index('Timestamp')
embassy_sensor = embassy_sensor.set_index('Date (LT)')

b21['PM 2.5'] = b21['PM 2.5']*b21['Applied PM 2.5 Custom Calibration Setting - Multiplication Factor']

#date range
start = pd.to_datetime(first_timestamp)
end = pd.to_datetime(last_timestamp)
dates = pd.date_range(start=start, end=end, freq='1H')

b21 = b21.loc[start:end+time_difference]
embassy_sensor = embassy_sensor.loc[start:end]
print(embassy_sensor['Raw Conc.'])
print(b21['PM 2.5'])

#data analysis ----------------------------------------------------------------
#fitting
def func(x, a, b):
    return a*x + b

x = embassy_sensor['Raw Conc.'].values
y = b21['PM 2.5'].values
cut1 = x >= 0
cut2 = y >= 0
cut = cut1 & cut2
x[~cut] = np.nan
y[~cut] = np.nan
popt, pcov = curve_fit(func, x[cut], y[cut], bounds=([-np.inf, -0.000001], [np.inf, 0.000001]))

figure = plt.figure(figsize=(10, 8))
plt.scatter(x[cut], y[cut])
plt.plot(x[cut], func(x[cut], *popt), color='red')
plt.ylabel('B21')
plt.xlabel('Embassy Sensor')
plt.axis('square')

#r2 value
residuals = y[cut] - func(x[cut], *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y[cut]-np.mean(y[cut]))**2)
r_squared = 1 - (ss_res / ss_tot)
r2_value = r_squared.round(2)

#nrmse
evaluator = RegressionMetric(y[cut], func(x[cut], *popt))
nrmse = evaluator.normalized_root_mean_square_error(model=2)*100

plt.text(0.3, 0.725, f"y={popt[0].round(2)}x", transform=figure.transFigure, fontsize=12)
plt.text(0.3, 0.7, f"R2={r2_value}", transform=figure.transFigure, fontsize=12)
plt.text(0.3, 0.675, f"NRMSE={nrmse.round(2)}%", transform=figure.transFigure, fontsize=12)
plt.title('B21 vs Embassy Sensor for values higher than 50')
plt.show()
#plt.savefig('B21_vs_embassy_sensor_fitting_higher_than_50.png')

#plotting
plt.figure(figsize=(15, 6))
plt.plot(b21['PM 2.5'], label='B21')
plt.plot(embassy_sensor['Raw Conc.'], label='Embassy Sensor')
plt.legend()
plt.ylim(0, 400)
plt.xlim(start, end)
plt.title('pre harmonization plot for values higher than 50')
plt.show()
#plt.savefig('pre_harmonization_plot_higher_than_50.png')

evaluator = RegressionMetric(y[cut], func(x[cut], *popt))
nrmse = evaluator.normalized_root_mean_square_error(model=2)  
print(nrmse)  

msd = np.sum((y[cut]-func(x[cut], *popt))**2)/len(y[cut])
rmsd = np.sqrt(msd)
print(rmsd)

print(rmsd/(np.max(y[cut])-np.min(y[cut])))


x_ = embassy_sensor.values
y_ = b21.values
cut1 = x_ >= 0
cut2 = y_ >= 0
cut = cut1 & cut2
#all
compare(x_[cut], y_[cut])

#lower than 50
cut1 = x_ < 0
cut2 = y_ < 50
cut = cut1 & cut2
compare(x_[cut], y_[cut], label='lower than 50')

#higher than 50
cut1 = x_ >= 50
cut2 = y_ >= 50
cut = cut1 & cut2
compare(x_[cut], y_[cut], label='higher than 50')