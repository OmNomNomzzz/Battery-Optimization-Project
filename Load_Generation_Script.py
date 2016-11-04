# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 14:28:04 2014

@author: Aniq Ahsan
"""
import csv
import scipy as sp
#import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import os
import pandas as pd
from numpy import *
from numpy.random import *
#import matplotlib.dates as md
from scipy import interpolate
from scipy import optimize
import time

"""
#Load File


"""
directory = r"D:\Battery Optimisation Project\Git Repository\Battery-Optimization-Project"#folder to save new file into


#a new decorator that vectorizes and makes the final value an array if it is an array of arrays.
def vectorize2(old_function, *args, **kwds):
    temp_function = vectorize(old_function, *args, **kwds)
    def new_function(*args2, **kwds2):
        return array(temp_function(*args2, **kwds2).tolist())
    return new_function

#3D plotting function#x,y,z vectors
def plot3d(*args):
    close()
    fig = figure()
    ax = fig.gca(projection = '3d')
    ax.plot_trisurf(*args,cmap=cm.jet,linewidth=0.2)
    show()



f = []
mypath = r"D:\Battery Optimisation Project\openei load data\RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT\RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT"#folder with all load data
for (dirpath,dirnames,filenames) in  walk(mypath):
    for filename in filenames:
        if filename.endswith(".csv"):
            filepath = os.path.join(dirpath,filename)
            f.append(filepath)

testload  = array([list(pd.read_csv(f[0])["Electricity:Facility [kW](Hourly)"])])
for counter1 in xrange(size(f)-1):
    tempdf = pd.read_csv(f[counter1 + 1])
    if ("Electricity:Facility [kW](Hourly)" in set(tempdf.columns)) & (tempdf.shape[0] == testload.shape[1]):
        testload = append(array([list(tempdf["Electricity:Facility [kW](Hourly)"])]),testload, axis = 0)
        

averageload = average(testload,axis = 0)

newfilename = directory + r"\loaddata.csv"
savetxt(newfilename,averageload, delimiter = ",")