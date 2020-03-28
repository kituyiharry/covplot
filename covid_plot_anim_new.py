#
# Author: Harry Kituyi Wakuloba
# covid_plot.py - plot covid-19 stats
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# import pandas as pd # Using numpy would require a lot of work!! ;)

# date processing
from datetime import datetime
import re

# Animation
from matplotlib.animation import FuncAnimation
from matplotlib import animation

# Date formatting
import matplotlib.dates as mdates

# from time import sleep
# import scipy.interpolate.make_interp_spline as spl

# Formaters for dates
days = mdates.DayLocator()      # every day
months = mdates.MonthLocator()  # every month
s_fmt = mdates.DateFormatter('%b-%Y')


# Printf debugging
def STAGE(stage, desc):
    print("[", stage, "]\t", desc)

def SUBSTAGE(substage, desc):
    print("\t[", substage, "]\t", desc, flush=True)

# Gives a human readable format of a number eg 1000000 = 1M
# refer to https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python
def human_format(num, _p):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

# All countries are ASCII strings
# Asks for input from the user for a country and tries to look it up
# Loops until we get one, CTRL-C exits
def askcountry(countryset, origdict, entry):
    prompt = "Enter a country [e.g. Mainland China, US is US]: "
    ctr = 0
    g = ""
    isfa = True # is unavailable
    while not g.lower() in  countryset:
        if ctr > 0 and isfa:
            prompt = g + " is not available. Try again\n" + prompt 
            isfa = False
        print(prompt, end='')
        g = str(input()).strip()
        ctr = ctr +1
    print(g + " has been chosen as {}, moving on".format(entry))
    return origdict[g.lower()]

def dateconv(date_str):
    """
    :returns: a date instance
    """
    # print("Cnv: ", date_str)
    return np.datetime64(datetime.strptime(date_str, '%m/%d/%y')) #ref: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior


STAGE("CSV-Read", "importing data")
# NB: Data needs to be presorted by date!
# then converting to numpy array
# np_data = np.recfromcsv('covid_19_clean_complete_with_hints.csv', delimiter=',', dtype=None, encoding="utf8",  names=True,
# indexes for easy indexing
# Also how i expect the data to be later on!
country, date, confirmed, deaths =  0,1,2,3 #,4

np_data = np.recfromcsv('covid_19_data.csv', delimiter=',', dtype=None, encoding="utf8",  names=True,
        converters={4: dateconv},
        usecols = (1,4,5,6),
        # usecols = (1,3,5,6,7),
        # missing_values={0: '???'},
        # filling_values={0: 'Unknown'}
    )   #

# Set comprehension
countries = {s[country].lower() for s in np_data}
ctrs      = dict([(s.lower(), s) for s in {s[country] for s in np_data} ])

# print(ctrs)

# print(askcountry(countries))

# print("US" in countries)


# Merge occurences based on the date as the may have
# occured on the same day in the same country in different days
def mergedata(data):
    STAGE("Merg-Data!", "Merging data from separate states in {}".format(data[0][country]))
    conf_accuml = 0
    dth_accuml  = 0
    rec_accuml  = 0
    ncases = []
    prev_date = data[0][date]
    counter = 0
    SUBSTAGE("Len data => ", len(data))
    # SUBSTAGE("First data => ", data[0][0], " at: ", data[0][1])
    # SUBSTAGE("Last data => ", data[len(data)-1][0], " at: ", data[len(data)-1][1])
    while counter < len(data):
        ctr,mdate,c,d = data[counter][country], data[counter][date], data[counter][confirmed],data[counter][deaths] #,data[counter][recovered]
        if counter == 0:
            # _,_,c,d,r = data[counter]
            # SUBSTAGE("Starting merger", "Accumulating values!! date already set => {}".format((c,d,r)))
            conf_accuml += c
            dth_accuml  += d
            # rec_accuml  += r
        elif mdate > prev_date :
            # SUBSTAGE("Mult-entry found", "Entry at =>  {}".format(counter))
            prevctr = data[(counter-1)][country]
            # print("Comparing dates!! => ", prev_date , " < ", mdate)
            # print("Appending for: => ", prevctr, "At: ", counter, " Currently: => ", len(ncases))
            # NB: Also swap these if the formatting changes!!
            ncases.append([
                prevctr, 
                prev_date,
                conf_accuml,
                dth_accuml,
                # rec_accuml,
            ])
            conf_accuml = c
            dth_accuml  = d
            # rec_accuml  = r
            prev_date   = mdate
        else:
            # print("Accumulating values!!", )
            conf_accuml += c
            dth_accuml  += d
            # rec_accuml  += r
        counter = counter+1
    SUBSTAGE("Finished", "Final shape => {}".format(len(ncases)))
    return ncases



# data = pd.read_csv('covid_19_data.csv', delimiter=',').to_numpy() # read data to numpy array
# data = pd.read_csv(
        # './covid_19_clean_complete.csv',  # file to open
        # delimiter=',',                    # csv files use , delimiter  
        # usecols=(1,4,5,6,7)               # 5 columns [Country, Date, Confirmed, Deaths, Recovered]
    # ) # read data to numpy array

# Display the data


#TODO: Use a region picker! from input
# Use set of all countries too!

STAGE("REGIONS", "Input a valid country\n\t **NB**: \n\t => China is Mainland China,\n\t => Multiple words should capitalize each Word! e.g Mainland China \n\t => Some countries are available as initials e.g US, UK...(Must all be capital!)")
region  = askcountry(countries, ctrs, "Region 1")
region2 = askcountry(countries, ctrs, "Region 2")
SUBSTAGE("Picked", "{} and {}".format(region, region2))

# newdata = mergedata(np_data)
# newdata = np_data

# print(newdata[1])
# Comprehension to get Cases only in Region
STAGE("Filtering", "filtering to selected regions")
cases   = np.array([s for s in np_data if s[country] == region])
SUBSTAGE(region, "Done!!")
cases_b = np.array([s for s in np_data if s[country] == region2])
SUBSTAGE(region2, "Done!!")

STAGE("MERGE!!", "starting merge!")
mergedcases = mergedata(cases)
mergedcases_b = mergedata(cases_b)

STAGE("MERGE", "Finished!")
print()
print(mergedcases[1])
print(mergedcases_b[1])
print()
# tconf = [case[confirmed] for case in merged]
# print(tconf)
# print(mergedata(cases_b))
# print(mergedata(cases))

STAGE("Helpers", "obtaining various helpers")

num_cases = len(mergedcases)
num_cases_b = len(mergedcases_b)

max_cases = np.max([num_cases, num_cases_b])
min_cases = np.min([num_cases, num_cases_b])
SUBSTAGE("CASES", "Max possible: {}, Min possible: {}".format(max_cases, min_cases))

def fmtnpdate(d):
    return str(d)[:10]

dates = np.array([case[date] for case in mergedcases], dtype='datetime64[D]')
dates_b = np.array([case[date] for case in mergedcases_b], dtype='datetime64[D]')
SUBSTAGE("DATES", "[{}] Max possible: {}, Min possible: {}".format(region, fmtnpdate(dates[0]), fmtnpdate(dates[len(dates)-1])))
SUBSTAGE("DATES", "[{}] Max possible: {}, Min possible: {}".format(region2, fmtnpdate(dates_b[0]), fmtnpdate(dates_b[len(dates_b)-1])))

STAGE("COMPUTE", "Deriving new data")

SUBSTAGE("CASES", "cumulative cases in {}".format(region))
cumulative_conf = np.cumsum([case[confirmed] for case in mergedcases])
cumulative_dth  = np.cumsum([case[deaths] for case in mergedcases])
# cumulative_rec  = np.cumsum([case[recovered] for case in mergedcases])

SUBSTAGE("CASES", "cumulative cases in {}".format(region2))
cumulconf_b = np.cumsum([case[confirmed] for case in mergedcases_b])
cumuldth_b  = np.cumsum([case[deaths] for case in mergedcases_b])
# cumulrec_b  = np.cumsum([case[recovered] for case in mergedcases_b])

# print(cumulconf_b[len(cumulconf_b)-10:len(cumulconf_b)])
# growth_a_vals = np.array(list(map(comp_growth, zip(range(len(cumulative_conf)), cumulative_conf))))
# growth_b_vals = np.array(list(map(comp_growth, zip(range(len(cumulconf_b)), cumulconf_b))))
# Does this give us the growth? refer to: 
#   https://stackoverflow.com/questions/24633618/what-does-numpy-gradient-do/24633888

# ges = np.gradient(cumulative_conf)
# gds = np.gradient(dates)

# print(dates[0], dates[1], dates[len(dates)-50])
# print("Gradients: ", gds, type(gds[0]))
# gbs = np.gradient(cumulconf_b)
SUBSTAGE("GRADIENTS", "");
growth_a_vals = np.gradient(cumulative_conf)
growth_b_vals = np.gradient(cumulconf_b)

# print("A vals: => ", growth_a_vals)
# print("B vals: => ", growth_b_vals)


print("Cases -> ", num_cases, "\tdates -> ", len(dates), "\nShapes\n conf: ", np.shape(cumulative_conf), "\tdths: ", np.shape(cumulative_dth), "\trec: ")
print("Cases[B] -> ", num_cases_b, "\tdates -> ", len(dates_b), "\nShapes\n conf: ", np.shape(cumulconf_b), "\tdths: ", np.shape(cumuldth_b), "\trec: ")

# Shapes should be similar !
print("Growth shapes: [A,B]: ", (np.shape(growth_a_vals), np.shape(growth_b_vals)))

# sleep(5)

STAGE("BUFFERS", "will be used to progressively animate data")
growth_a = []
growth_b = []
intbuffer  = []
dthbuffer  = []
# recbuffer  = []
datebuffer = []

int2buffer  = []
dth2buffer  = []
# rec2buffer  = []
date2buffer = []


STAGE("BOUNDS", "will be used to format the graph")
y_lim = np.max(cumulative_conf)
x_lim = np.max(dates)

y_lim_b = np.max(cumulconf_b)
x_lim_b = np.max(dates_b)

y_lim_ga = np.max(growth_a_vals)
y_lim_gb = np.max(growth_b_vals)

# print("y_lim_g is : ", [y_lim_ga, y_lim_gb])
# np.sort(dates)

# print(dates)
# Australian cases
# Cases may be less than dates!!

# plt.ion()

# plt.style.use('seaborn-whitegrid')

STAGE("PLOT", "Draw the damn thing!")
fig = plt.figure()
# plt.title("Covid-19 cases and growth\n " + region + " vs " + region2)

# g, x = plt.subplots(121)
# ax = g.subplots() 
# fig, axes = plt.subplots(1,2)

# print("[**]Figure Object :=> ", fig, "Axes :=> ", axes)
fig.set_figwidth(12.80)
fig.set_figheight(4.8)

#partition the axis
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122) 
# Plot inset
ax3 = fig.add_axes([0.13, 0.6, 0.1, 0.24]) 
ax4 = fig.add_axes([0.55, 0.6, 0.1, 0.24]) 
# print(dir(fig))
# Add a viewing area at the top that is 1week (7 days) of total cases for neatness
ax.set_xlim(dates[0], dates[len(dates)-1] + np.timedelta64(2,'D') )
# Add a viewing area at the top that is 1/4 of total cases for neatness
ax.set_ylim(0, y_lim+int(y_lim * 0.25))
line,  = ax.plot(datebuffer, intbuffer, 'b.', label='confirmed')
line2, = ax.plot(datebuffer, dthbuffer, 'r-', label='deaths')
# line3, = ax.plot(datebuffer, recbuffer, 'g-', label='recovered')

ax.grid(True)

#Setup legends
#
ax.legend([line, line2 ], [line.get_label(), line2.get_label()])

# print(dir(line))

# format the ticks
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(s_fmt)
ax.xaxis.set_minor_locator(days)

ax.set_title(region)
ax.set_xlabel("Date")
ax.set_ylabel("Cases")
ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(human_format)
)

# fig2, ax2 = plt.subplots(122)
# # Add a viewing area at the top that is 3 days of total cases for neatness
ax2.set_xlim(dates_b[0], dates_b[len(dates_b)-1] + np.timedelta64(3,'D') )
# # Add a viewing area at the top that is 1/4 of total cases for neatness
ax2.set_ylim(0, y_lim_b+int(y_lim_b * 0.125))
line4, = ax2.plot(date2buffer, int2buffer, 'b.', label='confirmed')
line5, = ax2.plot(date2buffer, dth2buffer, 'r-', label='deaths')
# line6, = ax2.plot(date2buffer, rec2buffer, 'g-', label='recovered')

ax2.grid(True)

# #Setup legends
# #
ax2.legend([line4, line5], [line4.get_label(), line5.get_label()])

# format the ticks
ax2.xaxis.set_major_locator(months)
ax2.xaxis.set_major_formatter(s_fmt)
ax2.xaxis.set_minor_locator(days)

ax2.set_title(region2)
ax2.set_xlabel("Date")
ax2.set_ylabel("Cases")
ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(human_format)
)

# plt.xlabel("Date")
# plt.ylabel("Cases")

# Add a viewing area at the top that is 1week (7 days) of total cases for neatness
ax3.set_xlim(dates[0], dates[len(dates)-1] + np.timedelta64(2,'D') )
ax4.set_xlim(dates_b[0], dates_b[len(dates_b)-1] + np.timedelta64(2,'D') )
# Add a viewing area at the top that is 1/4 of total cases for neatness
ax3.set_ylim(0, y_lim_ga+int(y_lim_ga * 0.33))
ax4.set_ylim(0, y_lim_gb+int(y_lim_gb * 0.33))

line7, = ax3.plot(datebuffer, growth_a, 'g-', label="Growth")
line8, = ax4.plot(date2buffer, growth_b, 'r-', label="Growth")

ax3.grid(False)
ax4.grid(False)

#Setup legends
#
ax3.legend([line7], [line7.get_label()])
ax4.legend([line8], [line8.get_label()])

# Hide dates
ax3.get_xaxis().set_visible(False)
ax4.get_xaxis().set_visible(False)

# Tick on the right side!
ax3.yaxis.tick_right()
ax4.yaxis.tick_right()

ax3.yaxis.set_major_formatter(
        ticker.FuncFormatter(human_format)
)
ax4.yaxis.set_major_formatter(
        ticker.FuncFormatter(human_format)
)
# print(dir(ax3.get_yaxis()))


# plt.xlabel("Date")
# plt.ylabel("Cases")
# print(dir(line))

# format the ticks
# ax3.xaxis.set_major_locator(months)
# ax3.xaxis.set_major_formatter(s_fmt)
# ax3.xaxis.set_minor_locator(days)
def init():
    print("Init called!!")
    line.set_data(datebuffer, intbuffer)
    line2.set_data(datebuffer, dthbuffer)
    # line3.set_data(datebuffer, recbuffer)
    line4.set_data(date2buffer, int2buffer)
    line5.set_data(date2buffer, dth2buffer)
    # line6.set_data(date2buffer, rec2buffer)
    line7.set_data(datebuffer, growth_a)
    line8.set_data(date2buffer, growth_b)
    return line, line2, line4, line5, line7, line8
    # return line, line2, line3, line4, line5, line6, line7, line8

comparator = num_cases > num_cases_b

def anim(i):
    # print("Frame: ", i)
    # num_cases is greater and probably max_cases => act normally
    # Naturally the shorter array will go faster
    if comparator:
        # print("num_cases is greater!!")
        # date2buffer should be less than min cases
        if len(date2buffer) < min_cases:
            date2buffer.append(dates_b[i])
            int2buffer.append(cumulconf_b[i])
            dth2buffer.append(cumuldth_b[i])
            # rec2buffer.append(cumulrec_b[i])
            growth_b.append(growth_b_vals[i])
            line4.set_data(date2buffer, int2buffer)
            line5.set_data(date2buffer, dth2buffer)
            # line6.set_data(date2buffer, rec2buffer)
            line8.set_data(date2buffer, growth_b)
        if len(datebuffer) < i+1:
            datebuffer.append(dates[i])
            intbuffer.append(cumulative_conf[i])
            dthbuffer.append(cumulative_dth[i])
            # recbuffer.append(cumulative_rec[i])
            growth_a.append(growth_a_vals[i])
            line.set_data(datebuffer, intbuffer)
            line2.set_data(datebuffer, dthbuffer)
            # line3.set_data(datebuffer, recbuffer)
            line7.set_data(datebuffer, growth_a)
        else:
            print("What")
    elif num_cases_b > num_cases:
        # print("num_cases_b is greater")
        if len(datebuffer) < min_cases:
            datebuffer.append(dates[i])
            intbuffer.append(cumulative_conf[i])
            dthbuffer.append(cumulative_dth[i])
            # recbuffer.append(cumulative_rec[i])
            growth_a.append(growth_a_vals[i])
            line.set_data(datebuffer, intbuffer)
            line2.set_data(datebuffer, dthbuffer)
            # line3.set_data(datebuffer, recbuffer)
            line7.set_data(datebuffer, growth_a)
        else:
            print("Ok")
        if len(date2buffer) < i+1:
            date2buffer.append(dates_b[i])
            int2buffer.append(cumulconf_b[i])
            dth2buffer.append(cumuldth_b[i])
            # rec2buffer.append(cumulrec_b[i])
            growth_b.append(growth_b_vals[i])
            line4.set_data(date2buffer, int2buffer)
            line5.set_data(date2buffer, dth2buffer)
            # line6.set_data(date2buffer, rec2buffer)
            line8.set_data(date2buffer, growth_b)
    else:
        # print("Theyre the same!!")
        datebuffer.append(dates[i])
        intbuffer.append(cumulative_conf[i])
        dthbuffer.append(cumulative_dth[i])
        # recbuffer.append(cumulative_rec[i])
        date2buffer.append(dates_b[i])
        int2buffer.append(cumulconf_b[i])
        dth2buffer.append(cumuldth_b[i])
        # rec2buffer.append(cumulrec_b[i])
        growth_a.append(growth_a_vals[i])
        growth_b.append(growth_b_vals[i])
        line4.set_data(date2buffer, int2buffer)
        line5.set_data(date2buffer, dth2buffer)
        # line6.set_data(date2buffer, rec2buffer)
        line8.set_data(date2buffer, growth_b)
        line.set_data(datebuffer, intbuffer)
        line2.set_data(datebuffer, dthbuffer)
        # line3.set_data(datebuffer, recbuffer)
        line7.set_data(datebuffer, growth_a)

    return line, line2, line4, line5, line7, line8
    # return line, line2, line3, line4, line5, line6, line7, line8

STAGE("ANIMATE", "animation starts!")

# Comment | Uncomment to save file!
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# animation = FuncAnimation(fig, anim, frames=range(num_cases), interval=1, blit=True, repeat=False)
anim = FuncAnimation(fig, anim,  init_func=init, frames=max_cases, interval=24, blit=True, repeat=False)
# anim.save('Sars-Cov-2 (' + region  ') vs (' + region2 + ').mp4', writer=writer)
# plt.yscale("log")
# plt.ioff()
plt.show()
