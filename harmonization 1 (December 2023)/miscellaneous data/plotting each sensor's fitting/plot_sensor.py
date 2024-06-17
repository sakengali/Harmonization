#finding the median values for each timestamp - 'parameter_data' is 2-dimensional arrays to store the values of the parameter for all sensors
parameter_data = [data[i][parameter].values for i in range(n_sensors)]

#cut off sensors with missing days to find the median in a simpler way
parameter_data = np.array([parameter_data[i] for i in range(n_sensors)])
median = np.nanmedian(parameter_data, axis=0)

x = median
y = data[0][parameter].values
        
def func(x, a, b):
        return a*x + b

#to omit values with nan
valid = ~ (np.isnan(x) | np.isnan(y))

#regression (the values are centralized to zero, by setting the intercept to zero)
popt, pcov = curve_fit(func, x[valid], y[valid], bounds=([-np.inf, -0.000001], [np.inf, 0.000001]))

plt.plot(median, data[0][parameter_name], 'o')
plt.plot(x, func(x, *popt), 'r--')
plt.axis('square')
plt.title(f"Sensor {sensor_names[0]}")
plt.xlabel("median")
plt.ylabel(parameter_name)
