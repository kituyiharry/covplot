#
# covid_plot.py - plot covid-19 stats
#

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd # Using numpy would require a lot of work!! ;)

# date processing
from datetime import datetime
import re

def dateconv(date_str):
    """
    :returns: a date instance
    """
    return np.datetime64(datetime.strptime(date_str, '%m/%d/%y')), #ref: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior

# This fails because of inconsistent data formatting. Resolving to using pandas instead
# then converting to numpy array
# np_data = np.recfromcsv('./covid_19_clean_complete.csv', delimiter=',', dtype=None, encoding="utf8",  names=True,
np_data = np.recfromcsv('covid_19_clean_complete_with_hints.csv', delimiter=',', dtype=None, encoding="utf8",  names=True,
            converters={4: dateconv, 5: int, 6: int, 7: int},
            # converters={5: int, 6: int, 7: int},
            usecols = (1,4,5,6,7),
            missing_values={0: '???'},
            filling_values={0: 'Unknown'}
        ) # 

# data = pd.read_csv('covid_19_data.csv', delimiter=',').to_numpy() # read data to numpy array
# data = pd.read_csv(
        # './covid_19_clean_complete.csv',  # file to open
        # delimiter=',',                    # csv files use , delimiter  
        # usecols=(1,4,5,6,7)               # 5 columns [Country, Date, Confirmed, Deaths, Recovered]
    # ) # read data to numpy array

# Display the data

# indexes
country, date, confirmed, deaths, recovered = 0,1,2,3,4

# Comprehension to get Cases only in Australia
aussie_cases = np.array([s for s in np_data if s[country] == "Australia"])
# aussie_cases = np_data

num_cases = len(aussie_cases)

cumulative_conf = np.zeros(num_cases, dtype=int)
cumulative_dth = np.zeros(num_cases, dtype=int)
cumulative_rec = np.zeros(num_cases, dtype=int)
log_conf = np.zeros(num_cases)
log_dth = np.zeros(num_cases)
log_rec = np.zeros(num_cases)

# dates = np.zeros(num_cases, dtype=int)
dates = np.linspace(0, num_cases, num_cases, dtype=int)
actual_dates = np.zeros(num_cases, dtype='datetime64[D]') # https://stackoverflow.com/questions/27469031/cannot-populate-numpy-datetime64-arrays
growth_rate = np.zeros(num_cases)
prev_case = 0

# Australian cases
print("Cases ->  ", num_cases)

index = 0
# TODO - Predict the iterations
for case in aussie_cases:
    # unpack case date
    # dates[index] = index 
    (actual_dates[index], ) = case[date] # Returns a tuple
    if index == 0:
        cumulative_conf[index] == case[confirmed]
        growth_rate[index] = 1
    else:
        cumulative_conf[index] = cumulative_conf[index-1] + case[confirmed]
        growth_rate[index] = case[confirmed] / prev_case
        prev_case = case[confirmed]
    prev_case = case[confirmed]
    index += 1

clean_growth = np.nan_to_num(growth_rate)
print(cumulative_conf)

# mean_growth = np.mean(clean_growth)
# max_growth = np.max(clean_growth)

plt.title("Covid-19")
# plt.yscale("log")
plt.plot(actual_dates, cumulative_conf, 'g')
plt.show()
