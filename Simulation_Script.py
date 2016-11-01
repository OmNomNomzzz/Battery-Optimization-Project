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
directory = r"D:\Battery Optimisation Project\Git Repository\Battery-Optimization-Project"
#battery data

#Lithium Ion
filenamev3 = directory + r"\Li1 Final Data.csv"
olddata3 = loadtxt(filenamev3,skiprows = 1, delimiter = ",")
pdata3 = olddata3[:,0]*1000.0#change to W
vdata3 = olddata3[:,1]
idata3 = -olddata3[:,2]
qdata3 = olddata3[:,3]


newfilename = directory + r"\Results"
newfilename2 = directory + r"\Results\simulationdata2.csv"
newfilename3 = directory + r"\Results\simulationdata3.csv"
newfilename4 = directory + r"\Results\simulationdata4.csv"






"""
#All universal constants are defined here
"""

#Units
#PowerUnits = W (total power of two DGs running at max capacity)
#TimeUnits = h
#CurrentUnits = A
#Charge = Ah
#VoltageUnits = V



#constants
startday = 1
endday = 30
dt = 5.0/60#0.5#timestep of simulation (hrs)
Dt =  5.0/60#0.5#dispatch period
n = 12
bounds = [(-1,1)]*n
acc = 1.0e-07
res = 40#resolution of graphs

#Li Ion Data
storageprice3 =50.0#dollars per kWh choose between 500-2000 for LA
capacity3 = 20#kWh per battery
battprice3 = storageprice3*capacity3
kp3 = 1.162#constant for Peukert Life unitless
dexp3 = 0.44#DOD for Peukert Life unitless
nexp3 = 1800.0#numer of cycles for Peukert Life unitless
iexp3 = 0.5#ratio for current dependence on degradation
Qmax3 = qdata3[175:200].max()*0.90#maximum charge, given by max discharge power curve
Qmin3 = qdata3[75:100].min()#Minimum Operational Charge of one Battery (Ah) given by max charge power curve
Pmaxc3 = -pdata3[75:100].min()#Maxmum Charging Power (W)
Pmaxd3 = pdata3[175:200].max()#Maxmum Discharging Power (W)
Imaxc3 = -idata3.min()
Imaxd3 = idata3.max()
n_batt3 = 900.0#number of batteries
bounds3 = [(-Pmaxc3,Pmaxd3)]*n
lcoe3 = storageprice3/(dexp3*nexp3)/1000.0#$/Wh
#DG Data
n_dg = 1.0
dgpmax = 1500000.0#Maximum DG power(W)
fuelprice = 2.54#S$/Gallon
#parameters of PV and load
loadfactor = 0.75#max load/ max dg
pvpenetration = 2.0#max pv/max load




"""
#Normalize Data
"""

filenamel = directory + r"\BaseSelectedAverage30min.csv"
oldloaddata = loadtxt(filenamel,skiprows = 0, delimiter = ",")
oldloaddata = oldloaddata[0:1488]
if Dt>0.5:
    loaddata = zeros(int(31*24/Dt))
    for x in xrange(loaddata.size):
        loaddata[x] = sum(oldloaddata[int(Dt/0.5)*x:int(Dt/0.5)*(x+1)])/int(Dt/0.5)
elif Dt == 0.5:
    loaddata = oldloaddata
else:
    loaddata = zeros(int(31*24/Dt))
    for x in xrange(loaddata.size):
        loaddata[x] = oldloaddata[int(Dt/0.5*x)]


filenamepv = directory + r"\PVData.csv"
oldpvdata = loadtxt(filenamepv,skiprows = 1, delimiter = ",")
pvdata = zeros(int(31*24/Dt))
if Dt>1.0/60.0:
    for x in xrange(pvdata.size):
        pvdata[x] = sum(oldpvdata[int(Dt*60)*x:int(Dt*60)*(x+1)])/int(Dt*60)


netload = loaddata/loaddata.max()*dgpmax*loadfactor - pvdata/pvdata.max()*dgpmax*loadfactor*pvpenetration




"""
#All new functions used are defined here
"""

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



#DG Data


def dgcost(P):
    p = P/(dgpmax*n_dg)
    if p<=0.0:
        return 0.0
    elif p<=0.01:
        return (-24750000000*p**5 + 1326.7*p)*fuelprice*n_dg*Dt#a*p**5+b*p
    elif p<=0.5:
        return (9.9 + 89.2*p)*fuelprice*n_dg*Dt
    elif p <= 0.75:
        return (7.9 + 93.2*p)*fuelprice*n_dg*Dt
    else:
        return (-8.3 + 114.8*p)*fuelprice*n_dg*Dt
dgcost_vec = vectorize2(dgcost)





#Li Ion
#Normalize data to 25 points only
'''
filenameli = r"D:\aniqahsan\My Documents\Work\Battery Testing\MGCC Data\Li1\Copy of Li1-20kW.csv"
olddata1 = loadtxt(filenameli,skiprows = 2, delimiter = ",")
newdata1 = zeros((25,4))
for x in xrange(25):
    #newdata1[x,:] = sum(olddata1[x*olddata1.shape[0]/25:(x+1)*olddata1.shape[0]/25],0)/(olddata1.shape[0]/25)
    newdata1[x,:] = olddata1[x*olddata1.shape[0]/25,:]
savetxt(r"D:\aniqahsan\My Documents\Work\Battery Testing\MGCC Data\Li1\Copy of Li1-20kW-2.csv",newdata1, delimiter = ",")
'''

pfun3 = sp.interpolate.LinearNDInterpolator(array([append(qdata3,linspace(0,Qmax3,25)),append(idata3,zeros(25))]).T,append(pdata3,zeros(25)))#add points so that power = 0 when current = 0.
#pfun3 = sp.interpolate.LinearNDInterpolator(array([qdata1,idata1]).T,pdata1)
pfun3_vec = vectorize2(pfun3)
ifun3 = sp.interpolate.LinearNDInterpolator(array([append(qdata3,linspace(0,Qmax3,25)),append(pdata3,zeros(25))]).T,append(idata3,zeros(25)))#add points so that power = 0 when current = 0.
ifun3_vec = vectorize2(ifun3)
vfun3 = sp.interpolate.LinearNDInterpolator(array([qdata3,idata3]).T,vdata3)
vfun3_vec = vectorize2(vfun3)
pminfun3 = sp.interpolate.interp1d(qdata3[75:100],pdata3[75:100],bounds_error = 0, fill_value = 0.0)
pminfun3_vec = vectorize2(pminfun3)
pmaxfun3 = sp.interpolate.interp1d(qdata3[175:200],pdata3[175:200],bounds_error = 0, fill_value = 0.0)
pmaxfun3_vec = vectorize2(pmaxfun3)
iminfun3 = sp.interpolate.interp1d(qdata3[175:200],idata3[175:200],bounds_error = 0, fill_value = 0.0)
iminfun3_vec = vectorize2(iminfun3)
imaxfun3 = sp.interpolate.interp1d(qdata3[75:100],idata3[75:100],bounds_error = 0, fill_value = 0.0)
imaxfun3_vec = vectorize2(imaxfun3)
def pmaxbatt3(q):
    if (q<Qmin3):
        return 0.0
    elif (q>Qmax3):
        return pmaxfun3(Qmax3)
    else:
        return pmaxfun3(q)
pmaxbatt3_vec = vectorize2(pmaxbatt3)
def pminbatt3(q):
    if (q<Qmin3):
        return pminfun3(Qmin3)
    elif (q>Qmax3):
        return 0.0
    else:
        return pminfun3(q)
pminbatt3_vec = vectorize2(pminbatt3)
def imaxd3(q):
    if (q<Qmin3):
        return 0.0
    elif (q>Qmax3):
        return -iminfun3(Qmax3)
    else:
        return min((q-Qmin3)/Dt,-iminfun3(q))
imaxd3_vec = vectorize2(imaxd3)
def imaxc3(q):
    if (q<Qmin3):
        return imaxfun3(Qmin3)
    elif (q>Qmax3):
        return 0.0
    else:
        return min((Qmax3-q)/Dt,imaxfun3(q))
imaxc3_vec = vectorize2(imaxc3)
def simbattp3(q,i):
    if i == 0:
        return 0.0
    if q<Qmin3:
        q1 = Qmin3
    elif q>Qmax3:
        q1 = Qmax3
    else:
        q1 = q
    minimumI = iminfun3(q1)
    maximumI = imaxfun3(q1)
    if i<minimumI:
        i1 = minimumI
    elif i>maximumI:
        i1 = maximumI
    else:
        i1 =i
    return pfun3(q1,i1)
simbattp3_vec = vectorize2(simbattp3)
def simbatti3(q1,p1):
    if p1==0.0:
        return 0.0
    if p1 > simbattp3(q1,-Imaxd3):
        return -Imaxd3
    elif p1 < simbattp3(q1,Imaxc3):
        return Imaxc3
    else:
        i = sp.optimize.brentq((lambda i: p1 - simbattp3(q1,i)),-Imaxd3,Imaxc3,disp = 0,maxiter = 1000)
    return i
simbatti3_vec = vectorize2(simbatti3)
def Nfail3(qf,i):#1/number of cycles to failure given final charge(normalized)
    qf = (qf>0)*qf
    return (255.5 - 1643.7 * log(1-qf/Qmax3))**(-1) *(i<=0)#*(1-iexp3*i/Imaxd3)
    #return (((1-qf/Qmax3)/dexp3)**kp3)/nexp3*(1-iexp3*i/Imaxd3)*(i<=0)
Nfail3_vec = vectorize2(Nfail3)


def battcost3(qi,P):#operation cost of using n_batt batteies to supply power P for time T at charge qi
    p = P/n_batt3    
    if (p <= 0.0)or(p>pmaxbatt3(qi)):
        return  nan
    i = zeros(Dt/dt+1)
    q = zeros(Dt/dt + 1)
    q[0] = qi
    for t in xrange(int(Dt/dt)):
        i[t] = simbatti3(q[t],p)
        q[t+1] = q[t] + i[t]*dt
    i[-1] = i[-2]
    if (sum(q<0)>0):
        return nan
    q = (q>0)*q
    cost = sum((Nfail3_vec(q[1:],i[:-1])-Nfail3(q[:-1],i[:-1]))*n_batt3*battprice3)
    return cost/(P*Dt)
battcost3_vec = vectorize2(battcost3)

def simbattcost3(i,qi,qf):#cost function for Aquion for simulation
    if sum(i<0)==0:
        return 0.0
    else:
        return sum((i<0)*(Nfail3_vec(qf,i)-Nfail3_vec(qi,i)))*battprice3

def indivbattcost3_vec(i,qi,qf):#cost function for Aquion for simulation
    return (i<0)*(Nfail3_vec(qf,i)-Nfail3_vec(qi,i))*battprice3


def totalcost3(i,q0,L):#Total cost function for Aquion for simulation
    qf = q0 + cumsum(i)*Dt
    qi = append(q0,qf[:-1])
    Pbatt = simbattp3_vec(qi,i)*n_batt3
    Pdg = ((L-Pbatt)>0)*(L-Pbatt)
    costdg = sum(dgcost_vec(Pdg))
    costbatt = sum(simbattcost3(i,qi,qf))*n_batt3
    return costdg + costbatt




def simbattcost4(i,qi,qf):#cost function for Aquion for simulation
    p = simbattp3_vec(qi,i)
    return sum((p>0)*p)*lcoe3*Dt

def indivbattcost4_vec(i,qi,qf):#cost function for Aquion for simulation
    p = simbattp3_vec(qi,i)
    return (p>0)*p*lcoe3*Dt

def totalcost4(i,q0,L):#Total cost function for Aquion for simulation
    qf = q0 + cumsum(i)*Dt
    qi = append(q0,qf[:-1])
    Pbatt = simbattp3_vec(qi,i)*n_batt3
    Pdg = ((L-Pbatt)>0)*(L-Pbatt)
    costdg = sum(dgcost_vec(Pdg))
    costbatt = sum(simbattcost4(i,qi,qf))*n_batt3
    return costdg + costbatt

"""
def simbattcost4(i,qi,qf):#cost function for Aquion for simulation
    if sum(i<0)==0:
        return 0.0
    else:
        return sum((i<0)*(Nfail3_vec(1-qf+qi,i)))*battprice3

def indivbattcost4_vec(i,qi,qf):#cost function for Aquion for simulation
    p = simbattp3_vec(qi,i)
    return (p>0)*p*lcoe3*Dt

def totalcost4(i,q0,L):#Total cost function for Aquion for simulation
    qf = q0 + cumsum(i)*Dt
    qi = append(q0,qf[:-1])
    Pbatt = simbattp3_vec(qi,i)*n_batt3
    Pdg = ((L-Pbatt)>0)*(L-Pbatt)
    costdg = sum(dgcost_vec(Pdg))
    costbatt = sum(simbattcost4(i,qi,qf))*n_batt3
    return costdg + costbatt
"""






def f_ieqcons3(i,q0,L):#inequatlity constraints for Aquion
    qf = q0 + cumsum(i)*Dt
    qi = append(q0,qf[:-1])
    Pbatt = simbattp3_vec(qi,i)*n_batt3
    Pdg = ((L-Pbatt)>0)*(L-Pbatt)
    return array((i+imaxd3_vec(qi)).tolist() + (imaxc3_vec(qi)-i).tolist() + (n_dg*dgpmax-Pdg).tolist() + (Pdg).tolist())



#Simulation



def spoint3(L,q0): #script to find a start point for simulation with Aquion
    x1 = zeros(2*n)#final answer, [Pdg,ibatt]
    qf = zeros(n+1)
    qf[0] = q0
    for t in xrange(n):
        if L[t]<0:#PV is more than Load
            x1[t] = 0.0
            i1 = imaxc3(qf[t])
            Pmaxb = simbattp3(qf[t],i1)*n_batt3
            if L[t] <= Pmaxb:
                x1[t+n] = i1
                qf[t+1] = qf[t]+i1*Dt
            else:
                x1[t+n] = simbatti3(qf[t],L[t]/n_batt3)
                qf[t+1] = qf[t] + x1[t+n]*Dt
        elif L[t]>0:#Load is more than PV
            i1 = -imaxd3(qf[t])
            Pmaxb = simbattp3(qf[t],i1)*n_batt3
            if L[t] >= Pmaxb:#Battery cannot handle load alone
                x1[t+n] = i1
                x1[t] = L[t] - Pmaxb
                qf[t+1] = qf[t] + i1*Dt
            else:#Battery can handle load alone
                x1[t] = 0.0
                x1[t+n] = simbatti3(qf[t],L[t]/n_batt3)
                qf[t+1] = qf[t] + x1[t+n]*Dt
        else:#load = 0
            qf[t+1] = qf[t]
    return x1

def simstep3(L,qi):
    if L[0]<0:
        i1 = imaxc3(qi)
        Pmaxb = simbattp3(qi,i1)*n_batt3
        if L[0] <= Pmaxb:
            i_f = i1
        else:
            i_f = simbatti3(qi,L[0]/n_batt3)
        q_f = qi+i_f*Dt
        Pbatt = simbattp3(qi,i_f)*n_batt3
        Pdg = 0.0
        Pnet = Pbatt-L[0]
        return array([q_f,Pbatt,Pdg,Pnet,i_f])
    x1 = spoint3(L,qi)[n:]
    x0 = zeros_like(x1)
    x = sp.optimize.fmin_slsqp(totalcost3, x1 ,f_ieqcons = f_ieqcons3, bounds = bounds3, args = (qi,L), disp = 0, full_output = 1, iter = n*10)
    if ((x[3]==2)or(x[3]==4)):#optimization failed
        cost0 = totalcost3(x0,qi,L)
        cost1 = totalcost3(x1,qi,L)
        if cost0 < cost1:#cheaper to not use battery at all
            #print "Exit Mode: %s choice = 0" % x[3]
            i = x0
        else:#spoint gives better result
            #print "Exit Mode: %s choice = 1" % x[3]
            i = x1
    else:#optimization successful
        cost0 = totalcost3(x0,qi,L)
        cost1 = totalcost3(x1,qi,L)
        costx = totalcost3(x[0],qi,L)
        #print "cost0: %s" % cost0
        #print "cost1: %s" % cost1
        #print "cost2: %s" % costx
        if ((costx<cost0)and(costx<cost1)):#optimization yields best result
            #print "Exit Mode: %s choice = 2" % x[3]
            i = x[0]
        elif (cost1<cost0):#spoint yields best result
            #print "Exit Mode: %s choice = 1" % x[3]
            i = x1
        else:#no battery usage is cheapest
            #print "Exit Mode: %s choice = 0" % x[3]
            i = x0
    q = qi+i[0]*Dt
    Pbatt = simbattp3(qi,i[0])*n_batt3
    Pdg = ((L[0]-Pbatt)>0)*(L[0]-Pbatt)
    Pnet = Pbatt+Pdg-L[0]
    return array([q,Pbatt,Pdg,Pnet,i[0]])




def fullsimulation3(netl):
    T = arange(netl.size-n)
    battqdata = zeros(T.size+1)
    battqdata[0] = (Qmax3-Qmin3)*0.0+Qmin3#rand()
    battpdata = zeros(T.size)
    dgpdata = zeros(T.size)
    pnetdata = zeros(T.size)
    battidata = zeros(T.size)
    counter = 1#counter
    t0 = time.time()
    for t in xrange(T.size):
        x = simstep3(netl[t:t+n],battqdata[t])
        battqdata[t+1] = x[0]
        battpdata[t] = x[1]
        dgpdata[t] = x[2]
        pnetdata[t] = x[3]
        battidata[t] = x[4]
        if counter == 24:
            savetxt(newfilename3, array([battqdata[1:], battpdata,dgpdata, pnetdata, battidata]).T, delimiter = ",")
            counter = 1
        else:
            counter +=1
        print "Time: %s" % t
    costdg = dgcost_vec(dgpdata)
    costbatt = indivbattcost3_vec(battidata,battqdata[:-1],battqdata[1:])*n_batt3
    costdata = costdg + costbatt
    avcost = sum(costdata)/(costdata.size)/Dt#average cost of operation
    t1 = time.time()
    dt = t1-t0
    y = array([battqdata[1:], battpdata,dgpdata, pnetdata, battidata, costdg, costbatt, costdata])
    savetxt(newfilename3, y.T, delimiter = ",")
    print "Red: Load minus PV"
    print (netl[:-n]/(dgpmax*n_dg)*100.0).round(2)
    print "Blue: Battery Power"
    print (battpdata/(dgpmax*n_dg)*100.0).round(2)
    print "Green: DG Power"
    print (dgpdata/(dgpmax*n_dg)*100.0).round(2)
    print "Light Blue: Net Load"
    print (-pnetdata/(dgpmax*n_dg)*100.0).round(2)
    print "Battery Current: %s" % (battidata/Imaxc3*100.0).round(2)
    print "Battery SOC: %s" % (battqdata/Qmax3*100.0).round(2)
    print "Time taken = %s" % dt
    print "Total Cost of DG is: SGD %s" % sum(costdg)
    print "Total Cost of Battery is: SGD %s" % sum(costbatt)
    print "Average Cost of operation per hour: SGD %s" % avcost
    battlife = storageprice3*capacity3*n_batt3*costdata.size/(sum(costbatt)*8760)
    print "Approximate life of Battery is: %s years" % battlife
    #close()
    #plot(T,battpdata/(dgpmax*n_dg)*100.0,T,dgpdata/(dgpmax*n_dg)*100.0,T,netl[:-n]/(dgpmax*n_dg)*100.0,T,(pnetdata)/(dgpmax*n_dg)*100.0,T,battqdata[:-1]/Qmax3*100.0)
    #legend(("Battery Power","DG Power","Load-PV","Power Curtailment","Battery SOC"))
    #show()
    return y



def simstep4(L,qi):
    if L[0]<0:
        i1 = imaxc3(qi)
        Pmaxb = simbattp3(qi,i1)*n_batt3
        if L[0] <= Pmaxb:
            i_f = i1
        else:
            i_f = simbatti3(qi,L[0]/n_batt3)
        q_f = qi+i_f*Dt
        Pbatt = simbattp3(qi,i_f)*n_batt3
        Pdg = 0.0
        Pnet = Pbatt-L[0]
        return array([q_f,Pbatt,Pdg,Pnet,i_f])
    x1 = spoint3(L,qi)[n:]
    x0 = zeros_like(x1)
    x = sp.optimize.fmin_slsqp(totalcost4, x1 ,f_ieqcons = f_ieqcons3, bounds = bounds3, args = (qi,L), disp = 0, full_output = 1, iter = n*10)
    if ((x[3]==2)or(x[3]==4)):#optimization failed
        cost0 = totalcost4(x0,qi,L)
        cost1 = totalcost4(x1,qi,L)
        if cost0 < cost1:#cheaper to not use battery at all
            #print "Exit Mode: %s choice = 0" % x[3]
            i = x0
        else:#spoint gives better result
            #print "Exit Mode: %s choice = 1" % x[3]
            i = x1
    else:#optimization successful
        cost0 = totalcost4(x0,qi,L)
        cost1 = totalcost4(x1,qi,L)
        costx = totalcost4(x[0],qi,L)
        #print "cost0: %s" % cost0
        #print "cost1: %s" % cost1
        #print "cost2: %s" % costx
        if ((costx<cost0)and(costx<cost1)):#optimization yields best result
            #print "Exit Mode: %s choice = 2" % x[3]
            i = x[0]
        elif (cost1<cost0):#spoint yields best result
            #print "Exit Mode: %s choice = 1" % x[3]
            i = x1
        else:#no battery usage is cheapest
            #print "Exit Mode: %s choice = 0" % x[3]
            i = x0
    q = qi+i[0]*Dt
    Pbatt = simbattp3(qi,i[0])*n_batt3
    Pdg = ((L[0]-Pbatt)>0)*(L[0]-Pbatt)
    Pnet = Pbatt+Pdg-L[0]
    return array([q,Pbatt,Pdg,Pnet,i[0]])




def fullsimulation4(netl):
    T = arange(netl.size-n)
    battqdata = zeros(T.size+1)
    battqdata[0] = (Qmax3-Qmin3)*0.0+Qmin3#rand()
    battpdata = zeros(T.size)
    dgpdata = zeros(T.size)
    pnetdata = zeros(T.size)
    battidata = zeros(T.size)
    counter = 1#counter
    t0 = time.time()
    for t in xrange(T.size):
        x = simstep4(netl[t:t+n],battqdata[t])
        battqdata[t+1] = x[0]
        battpdata[t] = x[1]
        dgpdata[t] = x[2]
        pnetdata[t] = x[3]
        battidata[t] = x[4]
        if counter == 24:
            savetxt(newfilename4, array([battqdata[1:], battpdata,dgpdata, pnetdata, battidata]).T, delimiter = ",")
            counter = 1
        else:
            counter +=1
        print "Time: %s" % t
    costdg = dgcost_vec(dgpdata)
    costbatt = indivbattcost4_vec(battidata,battqdata[:-1],battqdata[1:])*n_batt3
    actualcostbatt = indivbattcost3_vec(battidata,battqdata[:-1],battqdata[1:])*n_batt3
    costdata = costdg + costbatt
    actualcostdata = costdg + actualcostbatt
    avcost = sum(costdata)/(costdata.size)/Dt#average cost of operation
    actualavcost = sum(actualcostdata)/(actualcostdata.size)/Dt#average cost of operation
    t1 = time.time()
    dt = t1-t0
    y = array([battqdata[1:], battpdata,dgpdata, pnetdata, battidata, costdata, costdg, actualcostbatt, actualcostdata])
    savetxt(newfilename4, y.T, delimiter = ",")
    print "Red: Load minus PV"
    print (netl[:-n]/(dgpmax*n_dg)*100.0).round(2)
    print "Blue: Battery Power"
    print (battpdata/(dgpmax*n_dg)*100.0).round(2)
    print "Green: DG Power"
    print (dgpdata/(dgpmax*n_dg)*100.0).round(2)
    print "Light Blue: Net Load"
    print (-pnetdata/(dgpmax*n_dg)*100.0).round(2)
    print "Battery Current: %s" % (battidata/Imaxc3*100.0).round(2)
    print "Battery SOC: %s" % (battqdata/Qmax3*100.0).round(2)
    print "Time taken = %s" % dt
    print "Total Cost of DG is: SGD %s" % sum(costdg)
    print "Total Cost of Battery is: SGD %s" % sum(costbatt)
    print "Actual Total Cost of Battery is: SGD %s" % sum(actualcostbatt)
    print "Average Cost of operation per hour: SGD %s" % avcost
    print "Actual Average Cost of operation per hour: SGD %s" % actualavcost
    battlife = storageprice3*capacity3*n_batt3*costdata.size/(sum(costbatt)*8760)
    actualbattlife = storageprice3*capacity3*n_batt3*costdata.size/(sum(actualcostbatt)*8760)
    print "Approximate life of Battery is: %s years" % battlife
    print "Actual Approximate life of Battery is: %s years" % actualbattlife
    #close()
    #plot(T,battpdata/(dgpmax*n_dg)*100.0,T,dgpdata/(dgpmax*n_dg)*100.0,T,netl[:-n]/(dgpmax*n_dg)*100.0,T,(pnetdata)/(dgpmax*n_dg)*100.0,T,battqdata[:-1]/Qmax3*100.0)
    #legend(("Battery Power","DG Power","Load-PV","Power Curtailment","Battery SOC"))
    #show()
    return y




#Plotting Scripts






#Script to plot P data of n_batt1 Aquion Battery
def plotbattdataP3():
    mygrid = mgrid[0:Qmax3:res*1j,-Imaxd3:Imaxc3:res*1j]
    qdata = mygrid[0].flatten()
    idata = mygrid[1].flatten()
    fdata = -pfun3_vec(qdata,idata)/n_batt3
    ftest = delete(fdata,where((-(fdata>=1))*(-(fdata<1))))
    fmin = ftest.min()
    fmax = ftest.max()
    #close()
    fig = figure()
    ax = fig.gca(projection = '3d')
    surf = ax.plot_trisurf(qdata,idata,fdata,cmap=cm.jet,linewidth=0.1,antialiased=True,vmin = fmin, vmax = fmax)
    ax.set_xlabel('Charge(Ah)')
    ax.set_ylabel('Current (A)')
    ax.set_zlabel('Power (kW)')
    fig.colorbar(surf)
    #fig.suptitle('Aquion Battery Operational Power')
    show()




#Script to plot Cost data of n_batt1 Aquion Battery
def plotbattdataC3():
    mygrid = mgrid[7.0:Qmax3-1.0:res*1j,0:Pmaxd3*n_batt3:res*1j]
    qdata1_sim = mygrid[0].flatten()
    pdata1_sim = mygrid[1].flatten()
    cdata1_sim = battcost3_vec(qdata1_sim,pdata1_sim)*1000
    ctest = delete(cdata1_sim,where((-(cdata1_sim>=1))*(-(cdata1_sim<1))))
    cmin = ctest.min()
    cmax = ctest.max()
    #close()
    fig = figure()
    ax = fig.gca(projection = '3d')
    surf = ax.plot_trisurf(qdata1_sim,pdata1_sim/1000.0/n_batt3,cdata1_sim,cmap=cm.jet,linewidth=0.1,antialiased=True,vmin = cmin, vmax = cmax)
    ax.set_xlabel('Charge(Ah)')    
    ax.set_ylabel('Power (kW)')
    ax.set_zlabel('Operation Cost ($/kWh)')
    fig.colorbar(surf)
    #fig.suptitle('Aquion Battery Operational Cost')
    show()



#Plot maximum discharge power data for n_batt1 Aquion
def plotdgcost():
    pdata1 = linspace(dgpmax*0.1*n_dg,dgpmax*n_dg,1000)
    cdata1 = dgcost_vec(pdata1)*100000.0/(pdata1*Dt)
    #close()
    fig = figure()
    ax = fig.gca()
    ax.plot(pdata1/1000.0,cdata1)
    ax.set_xlabel('Power(kW)')
    ax.set_ylabel('Cost per energy (cents/kWh)')
    fig.suptitle('DG Operational Cost, '+ str(dgpmax/1000)+'kW Capacity, '+ str(fuelprice) + 'S$ per Gallon Fuel Price')    
    show()


#Plot maximum discharge power data for n_batt1 Aquion
def plotpmaxdata3():
    qdata1 = linspace(0,Qmax3,1000)
    pdata1 = pmaxbatt3_vec(qdata1)*n_batt3/1000.0
    #close()
    fig = figure()
    ax = fig.gca()
    ax.plot(qdata1,pdata1)
    ax.set_xlabel('Charge(Ah)')
    ax.set_ylabel('Maximum Power (kW)')
    fig.suptitle('Aquion Battery Maximum Discharging Power')    
    show()






#Plot maximum charge power data for n_batt2 LA
def plotpmindata3():
    qdata1 = linspace(0,Qmax3,1000)
    pdata1 = -pminbatt3_vec(qdata1)*n_batt3/1000.0
    #close()
    fig = figure()
    ax = fig.gca()
    ax.plot(qdata1,pdata1)
    ax.set_xlabel('Charge(Ah)')
    ax.set_ylabel('Maximum Power (kW)') 
    fig.suptitle('Lead Acid Battery Maximum Charging Power')
    show()



#Plot maximum current/max charging current  data for LA
def plotimaxdata3():
    qdata1 = linspace(0,Qmax3,1000)
    idata1 = imaxc3_vec(qdata1)
    #close()
    fig = figure()
    ax = fig.gca()
    ax.plot(qdata1,idata1)
    ax.set_xlabel('Charge(Ah)')
    ax.set_ylabel('Current (A)')
    fig.suptitle('Lead Acid Battery Maximum Charging Current')    
    show()

#Plot minimum current/max discharging current data for Aquion
def plotimindata3():
    qdata1 = linspace(0,Qmax3,1000)
    idata1 = imaxd3_vec(qdata1)
    #close()
    fig = figure()
    ax = fig.gca()
    ax.plot(qdata1,idata1)
    ax.set_xlabel('Charge(Ah)')
    ax.set_ylabel('Current (A)')    
    fig.suptitle('Aquion Battery Maximum Discharging Current')
    show()


"""
pvdaydatatype = [('day',int),('loadmax',float),('pvmax',float),('loadtotal',float),('pvtotal',float), ('powerdg1',float), ('costdg1',float), ('costbatt1',float), ('costtotal1',float), ('powerdg2',float), ('costdg2',float), ('costbatt2',float), ('costtotal2',float)]
for pv in xrange(1):
    pvpenetration = (pv + 1) * 0.4 + 2.0
    loadfactor = 0.88
    #finalload = append(zeros(34),loaddata)/loaddata.max()*dgpmax*loadfactor
    finalload = append(zeros(34),loaddata) * 500.0 * 1000.0
    #finalpv = append(zeros(34),pvdata)/pvdata.max()*dgpmax*loadfactor*pvpenetration
    finalpv = append(zeros(34),pvdata)/pvdata.max()*finalload.max()*pvpenetration
    netload = finalload - finalpv
    finalfilename = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\totalsimulationdata.csv"
    newfilename3 = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\simulationdata3.csv"
    newfilename4 = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\simulationdata4.csv"
    result3 = fullsimulation3(netload[int(startday*24/Dt):int(endday*24/Dt)+n].copy()).T#Simulate Lead Acid battery from startday to endday
    result4 = fullsimulation4(netload[int(startday*24/Dt):int(endday*24/Dt)+n].copy()).T#Simulate Lead Acid battery from startday to endday
    pvdaydata = array([(0,0,0,0,0,0,0,0,0,0,0,0,0)]*28,dtype = pvdaydatatype)
    for d in xrange(28):
        #pvpendata = finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)].max()/finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)].max()
        #pvratiodata = sum(finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)])/sum(finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)])
        pvdaydata[d] = (d+2, finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)].max(), finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)].max(), sum(finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)]), sum(finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)]), sum(result3[:,2][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result3[:,5][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result3[:,6][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result3[:,7][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result4[:,2][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result4[:,6][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result4[:,7][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result4[:,8][int((d+1)*24/Dt):int((d+2)*24/Dt)]))
    savetxt(finalfilename, pvdaydata, delimiter = ",")
"""


pvdaydatatype = [('day',int),('loadmax',float),('pvmax',float),('loadtotal',float),('pvtotal',float), ('powerdg1',float), ('costdg1',float), ('costbatt1',float), ('costtotal1',float), ('powerdg2',float), ('costdg2',float), ('costbatt2',float), ('costtotal2',float)]
pvpenetration = 1.6#(pv + 1) * 0.4 + 2.0
loadfactor = 0.88
#finalload = append(zeros(34),loaddata)/loaddata.max()*dgpmax*loadfactor
finalload = append(zeros(34),loaddata) * 500.0 * 1000.0
#finalpv = append(zeros(34),pvdata)/pvdata.max()*dgpmax*loadfactor*pvpenetration
finalpv = append(zeros(34),pvdata)/pvdata.max()*finalload.max()*pvpenetration
netload = finalload - finalpv

for bp in xrange(1):
    storageprice3 =(bp+1)*25+25#dollars per kWh choose between 500-2000 for LA
    battprice3 = storageprice3*capacity3
    lcoe3 = storageprice3/(dexp3*nexp3)/1000.0#$/Wh
    finalfilename = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\battprice_" + str(storageprice3) + r"\totalsimulationdata.csv"
    newfilename3 = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\battprice_" + str(storageprice3) + r"\simulationdata3.csv"
    newfilename4 = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\battprice_" + str(storageprice3) + r"\simulationdata4.csv"
    result3 = fullsimulation3(netload[int(startday*24/Dt):int(endday*24/Dt)+n].copy()).T#Simulate Lead Acid battery from startday to endday
    result4 = fullsimulation4(netload[int(startday*24/Dt):int(endday*24/Dt)+n].copy()).T#Simulate Lead Acid battery from startday to endday
    pvdaydata = array([(0,0,0,0,0,0,0,0,0,0,0,0,0)]*28,dtype = pvdaydatatype)
    for d in xrange(28):
        #pvpendata = finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)].max()/finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)].max()
        #pvratiodata = sum(finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)])/sum(finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)])
        pvdaydata[d] = (d+2, finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)].max(), finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)].max(), sum(finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)]), sum(finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)]), sum(result3[:,2][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result3[:,5][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result3[:,6][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result3[:,7][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result4[:,2][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result4[:,6][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result4[:,7][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result4[:,8][int((d+1)*24/Dt):int((d+2)*24/Dt)]))
    savetxt(finalfilename, pvdaydata, delimiter = ",")


#different min SOC
"""
pvdaydatatype = [('day',int),('loadmax',float),('pvmax',float),('loadtotal',float),('pvtotal',float), ('powerdg1',float), ('costdg1',float), ('costbatt1',float), ('costtotal1',float), ('powerdg2',float), ('costdg2',float), ('costbatt2',float), ('costtotal2',float)]
pvpenetration = 2.4#(pv + 1) * 0.4 + 2.0
Qmindata = ones(8) * 28.952 #array([10,15,20,46.5,46.5,46.5,46.5,46.5])
loadfactor = 0.88
#finalload = append(zeros(34),loaddata)/loaddata.max()*dgpmax*loadfactor
finalload = append(zeros(34),loaddata) * 500.0 * 1000.0
#finalpv = append(zeros(34),pvdata)/pvdata.max()*dgpmax*loadfactor*pvpenetration
finalpv = append(zeros(34),pvdata)/pvdata.max()*finalload.max()*pvpenetration
netload = finalload - finalpv


for bp in xrange(8):
    bp2 = bp
    storageprice3 =(bp2+1)*25#dollars per kWh choose between 500-2000 for LA
    Qmin3 = Qmindata[bp2]
    battprice3 = storageprice3*capacity3
    lcoe3 = storageprice3/(dexp3*nexp3)/1000.0#$/Wh
    finalfilename = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\battprice_" + str(storageprice3) + r"\totalsimulationdata_new2.csv"
    newfilename3 = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\battprice_" + str(storageprice3) + r"\simulationdata3.csv"
    newfilename4 = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\battprice_" + str(storageprice3) + r"\simulationdata4_new2.csv"
    #result3 = fullsimulation3(netload[int(startday*24/Dt):int(endday*24/Dt)+n].copy()).T#Simulate Lead Acid battery from startday to endday
    result3 = loadtxt(newfilename3,skiprows = 0, delimiter = ",")
    result4 = fullsimulation4(netload[int(startday*24/Dt):int(endday*24/Dt)+n].copy()).T#Simulate Lead Acid battery from startday to endday
    pvdaydata = array([(0,0,0,0,0,0,0,0,0,0,0,0,0)]*28,dtype = pvdaydatatype)
    for d in xrange(28):
        #pvpendata = finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)].max()/finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)].max()
        #pvratiodata = sum(finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)])/sum(finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)])
        pvdaydata[d] = (d+2, finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)].max(), finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)].max(), sum(finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)]), sum(finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)]), sum(result3[:,2][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result3[:,5][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result3[:,6][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result3[:,7][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result4[:,2][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result4[:,6][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result4[:,7][int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(result4[:,8][int((d+1)*24/Dt):int((d+2)*24/Dt)]))
    savetxt(finalfilename, pvdaydata, delimiter = ",")
"""



"""
pvdaydatatype2 = [('day',int),('pvtotal',float), ('costtotalop1',float), ('costtotallcoe1',float), ('costtotalop2',float), ('costtotallcoe2',float)]
for pv in xrange(1):
    pvpenetration = (pv + 1) * 0.4 + 2.4
    loadfactor = 0.88
    #finalload = append(zeros(34),loaddata)/loaddata.max()*dgpmax*loadfactor
    finalload = append(zeros(34),loaddata) * 500.0 * 1000.0
    #finalpv = append(zeros(34),pvdata)/pvdata.max()*dgpmax*loadfactor*pvpenetration
    finalpv = append(zeros(34),pvdata)/pvdata.max()*finalload.max()*pvpenetration
    netload = finalload - finalpv
    finalfilename = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\totalsimulationdata.csv"
    newfilename3 = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\simulationdata3.csv"
    newfilename4 = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\simulationdata4.csv"
    finalfilename2 = directory + r"\Results"+ r"\batt_"+ str(n_batt3)+ r"\load_" + str(loadfactor) + r"pv_" + str(pvpenetration) + r"\totalsimulationdata_costs.csv"
    results3 = loadtxt(newfilename3,skiprows = 0, delimiter = ",").T
    results4 = loadtxt(newfilename4,skiprows = 0, delimiter = ",").T
    #restults3: Simulation with dynamical cost
    battqdata3 = append(0.0,results3[0])
    battidata3 = results3[4]
    costdg3 = results3[5]
    battlcoecost3 = indivbattcost4_vec(battidata3,battqdata3[:-1],battqdata3[1:])*n_batt3
    battopcost3 = indivbattcost3_vec(battidata3,battqdata3[:-1],battqdata3[1:])*n_batt3
    lcoecostdata3 = costdg3 + battlcoecost3#total cost of operation calulated using lcoe
    opcostdata3 = costdg3 + battopcost3#total cost of operation calculated using dynamical cost
    #restults4: Simulation with lcoe cost
    battqdata4 = append(0.0,results4[0])
    battidata4 = results4[4]
    costdg4 = results4[6]
    battlcoecost4 = indivbattcost4_vec(battidata4,battqdata4[:-1],battqdata4[1:])*n_batt3
    battopcost4 = indivbattcost3_vec(battidata4,battqdata4[:-1],battqdata4[1:])*n_batt3
    lcoecostdata4 = costdg4 + battlcoecost4#total cost of operation calulated using lcoe
    opcostdata4 = costdg4 + battopcost4#total cost of operation calculated using dynamical cost
    pvdaydata = array([(0,0,0,0,0,0)]*28,dtype = pvdaydatatype2)
    for d in xrange(28):
        #pvpendata = finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)].max()/finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)].max()
        #pvratiodata = sum(finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)])/sum(finalload[int((d+2)*24/Dt):int((d+3)*24/Dt)])
        pvdaydata[d] = (d+2, sum(finalpv[int((d+2)*24/Dt):int((d+3)*24/Dt)]), sum(opcostdata3[int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(lcoecostdata3[int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(opcostdata4[int((d+1)*24/Dt):int((d+2)*24/Dt)]), sum(lcoecostdata4[int((d+1)*24/Dt):int((d+2)*24/Dt)]))
    savetxt(finalfilename2, pvdaydata, delimiter = ",")
"""
#Simulation script. Uncomment to run required simulation


#result3 = fullsimulation3(netload[int(startday*24/Dt):int(endday*24/Dt)+n].copy())#Simulate Lead Acid battery from startday to endday
#result4 = fullsimulation4(netload[int(startday*24/Dt):int(endday*24/Dt)+n].copy())#Simulate Lead Acid battery from startday to endday
#costdiff = (sum(result4[6])-sum(result3[5]))/(sum(result3[5]))*100.0
#print "cost difference between operational plans = %s percent" % costdiff
'''
data3 = loadtxt(r"C:\Users\Aniq\Documents\EPGC\Work\Battery Optimisation Project\Final Simulation\Results\pv600_l20_pr750\simulationdata3.csv", skiprows=0, delimiter = ",")
data4 = loadtxt(r"C:\Users\Aniq\Documents\EPGC\Work\Battery Optimisation Project\Final Simulation\Results\pv600_l20_pr750\simulationdata4.csv", skiprows=0, delimiter = ",")
pvdaydatatype = [('day',int),('pv',float),('cost1',float),('cost2',float)]
pvdaydata = array([(0,0,0,0)]*28,dtype = pvdaydatatype)
for d in xrange(28):
    #pvdaydata[d] = (d,sum(pvdata[int((d+2)*24/Dt):int((d+3)*24/Dt)]))
    pvdaydata[d] = (d+2,sum(pvdata[int((d+2)*24/Dt):int((d+3)*24/Dt)])/sum(loaddata[int((d+2)*24/Dt):int((d+3)*24/Dt)])*600.0/loadsize,sum(data3[:,-1][int((d+1)*24/Dt):int((d+2)*24/Dt)]),sum(data4[:,-1][int((d+1)*24/Dt):int((d+2)*24/Dt)]))
daydata = sort(pvdaydata,order = 'pv')
savetxt(newfilename, daydata, delimiter = ",")
'''
"""
mydata = loadtxt(r"C:\Users\Aniq\Documents\EPGC\Work\Battery Optimisation Project\RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT\BaseSelectedAverage.csv", skiprows=1, delimiter = ",", usecols = {11})
mydata2 = zeros(mydata.size*2)
for d in xrange(mydata.size*2):
    if d%2 == 0:
        mydata2[d] = mydata[d/2]
    elif d<mydata.size*2 - 1:
        mydata2[d] = (mydata[(d-1)/2] + mydata[(d+1)/2])/2.0
    else:
        mydata2[d] = mydata[(d-1)/2]

savetxt(r"C:\Users\Aniq\Documents\EPGC\Work\Battery Optimisation Project\RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT\BaseSelectedAverage30min.csv", mydata2, delimiter = ",")
"""