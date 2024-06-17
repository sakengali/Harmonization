%matplotlib inline
sensor_ind = 0
parameter_name = parameters[1]
sensor_harmonized_data = pd.read_csv(f'{result_folder}/{level2_folder}/{sensor_names[sensor_ind]}.csv')[parameter_name].values

tempp = [abs(i.round(2)) for i in sensor_harmonized_data - median]

tempp = np.where([i > 2 for i in tempp])
tempp
#plt.plot(dates[tempp], data[sensor_ind][parameter_name].values[tempp], label=f'sensor {sensor_names[sensor_ind]}')
#plt.plot(dates[tempp], median[tempp], label='median')

#outlier timestamps - ind 7, 8 
