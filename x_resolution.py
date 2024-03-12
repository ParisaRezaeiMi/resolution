#from sys import stdlib_module_names
from math import log
import pandas as pd
import sqlite3
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from brokenaxes import brokenaxes

os.system('rm -rf nx_ny_small')

data = pd.read_sql(
    'SELECT `n_channel`,`n_trigger`,`n_position`,`t_50 (s)`, `Amplitude (V)` FROM dataframe_table WHERE n_pulse==1',
    con=sqlite3.connect("parsed_from_waveforms.sqlite"),
)
positions_data = pd.read_pickle('positions.pickle')
positions_data.reset_index(['n_x','n_y'], drop=False, inplace=True)
data.set_index('n_position', inplace=True) #sets n_position as index
#since n_position is an index for both datasets, we can merge them
data = pd.merge(data, positions_data, left_index=True, right_index=True, how='inner')
#makes the n_position back as a data, so resets index and makes sure to not drop n_position
data.reset_index('n_position', inplace=True,drop=False)
data12 = data[(data['n_channel'] == 1) | (data['n_channel'] == 2)]
data12 = data[data['n_channel'].isin([1, 2])]



data_bad= data12.query('`Amplitude (V)` < 0.01')

# Create a boolean mask for rows to be dropped based on matching 'n_position' and 'n_trigger'
mask = data12.set_index(['n_position', 'n_trigger']).index.isin(data_bad.set_index(['n_position', 'n_trigger']).index)
# Keep only the rows that are not in bad_data
filtered_data = data12[~mask]



###### Plot generating - delta t_50 of ch1 and 2 vs X at a certain y
n_y = 15
horizontal_data_1 = filtered_data.query(f'`n_y`=={n_y} & n_channel == 1')
horizontal_data_2 = filtered_data.query(f'`n_y`=={n_y} & n_channel == 2')
# To be able to subtract the t_50 column, we need to set a matching index
delta_t_data = filtered_data.query(f'`n_y`=={n_y} & n_channel == 2')
delta_t_data = delta_t_data.reset_index()
horizontal_data_1 = horizontal_data_1.reset_index()

delta_t_data['t_50 (s)'] =   horizontal_data_1['t_50 (s)'] - delta_t_data['t_50 (s)']
sd_delta_t_1 = delta_t_data.groupby(['n_position']).std(ddof=0)
sd_delta_t_1_np = sd_delta_t_1['t_50 (s)'].to_numpy()

Avg_delta_t = delta_t_data.groupby(['n_position']).agg(np.nanmedian)
Avg_delta_t_np = Avg_delta_t['t_50 (s)'].to_numpy()
X_ch12_np= Avg_delta_t['x (m)'].to_numpy()

#plt.errorbar((X_ch12_np-np.mean(X_ch12_np))*1e6, Avg_delta_t_np, yerr= sd_delta_t_1_np, fmt='o', markersize=2)
#plt.xlabel('X_ch12 (m)')
#plt.ylabel('delta_t_12')
#plt.title('delta_t vs X AND `n_y`==15, voltage cut at 0.01 V')
#plt.legend()
#plt.show()

d_delta_t = np.diff(Avg_delta_t_np)
d_x=np.diff(X_ch12_np)
grad_delta_t = d_delta_t/d_x

#plt.scatter(((X_ch12_np-np.mean(X_ch12_np))*1e6)[:-1], grad_delta_t, label='grad delta t', s=2)
#plt.xlabel('X_ch12 (m)')
#plt.ylabel('delta_t_12')
#plt.title('delta_t vs X AND' f'`n_y`=={n_y}'', voltage cut at 0.01 V')
#plt.legend()
#plt.show()
###removing the outliers
delta_x_12 = np.absolute((sd_delta_t_1_np[:-1]/grad_delta_t)*1e6)
X_ch12_np = ((X_ch12_np - np.mean(X_ch12_np)) * 1e6)[:-1]
# Set the threshold
threshold = 200
# Identify indices where the first array exceeds the threshold
indices_to_remove = np.where(delta_x_12 > threshold)[0]
# Remove corresponding elements from both arrays
delta_x_12 = np.delete(delta_x_12, indices_to_remove)
X_ch12_np_filtered = np.delete(X_ch12_np, indices_to_remove)

#plt.scatter(X_ch12_np_filtered,delta_x_12, label='delta x using t', s=2)
#plt.scatter(((X_ch12_np-np.mean(X_ch12_np))*1e6)[:-1],delta_x_12 , label='delta x using t', s=2)
#plt.yscale('log')
#plt.xlabel('X_ch12 (um)')
#plt.ylabel('delta X_ch12 (um)')
#plt.title('delta_t vs X AND' f'`n_y`=={n_y}'', voltage cut at 0.01 V')
#plt.legend()
#plt.show()


# Calculate the regression line
coefficients = np.polyfit(X_ch12_np_filtered,
                          delta_x_12,
                          deg=1)

# Generate the regression line using the coefficients
regression_line = np.poly1d(coefficients)

# Scatter plot
#plt.scatter(X_ch12_np_filtered,
#            delta_x_12,
#            label='delta x using t', s=2)

# Plot the regression line
#plt.plot(X_ch12_np_filtered, regression_line(X_ch12_np_filtered),
#         color='red', label='Regression Line')
#plt.yscale('log')
#plt.xlabel('X_ch12 (um)')
#plt.ylabel('delta X_ch12 (um)')
#plt.title(f'delta_t vs X, n_y={n_y}, voltage cut at 0.01 V')
#plt.legend()
#plt.show()

###### Plot generating - delta Voltage of ch1 vs X at a certain y
horizontal_data_1 = filtered_data.query('`n_y`==15 & n_channel == 1')
Avg_hor_data_1 = horizontal_data_1.groupby(['n_position']).agg(np.nanmedian)
Amp_st_data_1 = horizontal_data_1.groupby(['n_position']).std(ddof=0)
horizontal_V_1 = Avg_hor_data_1['Amplitude (V)'].to_numpy()
amp_st_np = Amp_st_data_1['Amplitude (V)'].to_numpy()
X_ch12_np = Avg_hor_data_1['x (m)'].to_numpy()
d_V=np.diff(horizontal_V_1)
d_x=np.diff(X_ch12_np)
grad_V = d_V/d_x

#plt.scatter((X_ch12_np-np.mean(X_ch12_np))*1e6, horizontal_V_1, label='V-X', s=2)
#plt.scatter((X_ch12_np-np.mean(X_ch12_np))*1e6, amp_st_np, label='standard deviation', s=2, color = 'red')
#plt.errorbar((X_ch12_np-np.mean(X_ch12_np))*1e6, horizontal_V_1, yerr= amp_st_np, fmt='o')
#plt.xlabel('X_ch12_np')
#plt.ylabel('normalized_V_1')
#plt.title('V vs X AND `n_y`==15, voltage cut at 0.01 V')
#plt.legend()
#plt.show()


#plt.scatter(((X_ch12_np-np.mean(X_ch12_np))*1e6)[:-1], grad_V, label='gradient', s=2, color = 'red')
#plt.xlabel('X_ch12_np')
#plt.ylabel('gradient')
#plt.title('V vs X AND `n_y`==15, voltage cut at 0.01 V')
#plt.legend()
#plt.show()


#plt.scatter(((X_ch12_np-np.mean(X_ch12_np))*1e6)[:-1],(amp_st_np[:-1]/grad_V)*1e6 , label='gradient', s=2, color = 'red')
#(amp_st_np[:-1]/grad_V)*1e6
#plt.xlabel('X_ch12_np')
#plt.ylabel('delta x')
#plt.title('V vs X AND `n_y`==15, voltage cut at 0.01 V')
#plt.legend()
#plt.show()



#This function plots the x resolution vs x axis using delta_t_12

def x_res_t ( n_y_arr):
    for i in n_y_arr:
        n_y = i
        horizontal_data_1 = filtered_data.query(f'`n_y`=={n_y} & n_channel == 1')
        horizontal_data_2 = filtered_data.query(f'`n_y`=={n_y} & n_channel == 2')
        # To be able to subtract the t_50 column, we need to set a matching index
        delta_t_data = filtered_data.query(f'`n_y`=={n_y} & n_channel == 2')
        delta_t_data = delta_t_data.reset_index()
        horizontal_data_1 = horizontal_data_1.reset_index()

        delta_t_data['t_50 (s)'] =   horizontal_data_1['t_50 (s)'] - delta_t_data['t_50 (s)']
        sd_delta_t_1 = delta_t_data.groupby(['n_position']).std(ddof=0)
        sd_delta_t_1_np = sd_delta_t_1['t_50 (s)'].to_numpy()

        Avg_delta_t = delta_t_data.groupby(['n_position']).agg(np.nanmedian)
        Avg_delta_t_np = Avg_delta_t['t_50 (s)'].to_numpy()

        X_ch12_np= Avg_delta_t['x (m)'].to_numpy()
        d_delta_t = np.diff(Avg_delta_t_np)
        d_x=np.diff(X_ch12_np)
        grad_delta_t = d_delta_t/d_x
        plt.plot(((X_ch12_np-np.mean(X_ch12_np))*1e6)[:-1],np.absolute((sd_delta_t_1_np[:-1]/grad_delta_t)*1e6), label= f'`n_y`=={n_y}')
        plt.yscale('log')
        plt.xlabel('X_ch12 (um)')
        plt.ylabel('delta X_ch12 (um)')
        plt.title('delta_t vs X, voltage cut at 0.01 V')
        plt.legend()
    return (plt.show())    


        



#This function plots the x resolution vs x axis using amplitude V
def x_res_v ( n_y_arr):
    for i in n_y_arr:
        n_y = i
        horizontal_data_1 = filtered_data.query(f'`n_y`=={n_y} & n_channel == 1')
        Avg_hor_data_1 = horizontal_data_1.groupby(['n_position']).agg(np.nanmedian)
        Amp_st_data_1 = horizontal_data_1.groupby(['n_position']).std(ddof=0)
        # To be able to subtract the t_50 column, we need to set a matching index
        horizontal_V_1 = Avg_hor_data_1['Amplitude (V)'].to_numpy()

        amp_st_np = Amp_st_data_1['Amplitude (V)'].to_numpy()
        
        horizontal_data_1 = horizontal_data_1.reset_index()
        X_ch12_np = Avg_hor_data_1['x (m)'].to_numpy()


        d_V=np.diff(horizontal_V_1)
        d_x=np.diff(X_ch12_np)
        grad_V = d_V/d_x
        #plt.scatter(((X_ch12_np-np.mean(X_ch12_np))*1e6)[:-1],(amp_st_np[:-1]/grad_V)*1e6 ,  label= f'`n_y`=={n_y}', s=2)
        plt.plot(((X_ch12_np-np.mean(X_ch12_np))*1e6)[:-1],(amp_st_np[:-1]/grad_V)*1e6 ,  label= f'`n_y`=={n_y}')
        plt.yscale('log')
        plt.xlabel('X_ch12 (um)')
        plt.ylabel('delta X_ch12 (um)')
        plt.title('delta_x vs X, using V, voltage cut at 0.01 V')
        plt.legend()
    return (plt.show())    


n_y_arr = [5,15,18, 25,35,  45]
x_res_v ( n_y_arr)
x_res_t ( n_y_arr)

















