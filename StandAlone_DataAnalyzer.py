#import ROOT

import sys
import math

import numpy as np
#from ROOT import TH1D,TF1, gRandom, gPad, gStyle, TCanvas
import matplotlib as mpl
import matplotlib.pyplot as plt
#from ROOT import TChain, TSelector, TTree
import os
import matplotlib.colors as mcolors
import scipy
from matplotlib.colors import LogNorm
import pandas as pd
import seaborn as sns
#%matplotlib inline
import matplotlib.colors as colors
from scipy.optimize import curve_fit
import scipy.signal
from lmfit.models import GaussianModel, ConstantModel
from datetime import datetime
from glob import glob
import argparse
import shutil
import pylandau



mpl.rcParams.update(mpl.rcParamsDefault)
current_palette = sns.color_palette('colorblind', 10)
#import warnings
#warnings.filterwarnings("ignore")


from dt5202 import dt5202_crunch

from dt5202oldDAQ import dt5202_crunch as dt5202_crunch_oldDAQ
ped = 50 #pedestal correction


def parseData(fpath,chan):
    print (type(chan))
    if (os.path.isdir(fpath)):
        sourceFolder = fpath
        onlyfiles = [f for f in os.listdir(sourceFolder) if '.dat' in f]
        data_list = []
        for file in onlyfiles:
            data_list.append(dt5202_crunch(str(sourceFolder)+str(file), num_ev=10000000, num_ch=int(chan)))
        
        data = data_list[0]
        for i in range(1, len(data_list)):
            data = np.append(data, data_list[i])
            
        return data
    
    if os.path.isfile(fpath):
        data = dt52052_crunch(fpath,num_ev=10000000, num_ch=chan)
        return data
    
    
def gauss(x, mu, sigma):
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))


def sps(x, ped = 50, gain = 50, width_base = 5, width_scale = 1, poisson_k = 5, output_single_peaks = False):

    out = np.zeros_like(x)

    out_peaks = []

    for n_pe in np.arange(0, int((3*poisson_k)+5), 1):
        
        A = scipy.stats.poisson.pmf(n_pe, poisson_k)
        mu = ped + gain * n_pe
        sigma = np.sqrt(width_base**2 +  width_scale**2 * n_pe)

        peak = A * gauss(x, mu, sigma)
        
        out += peak
        out_peaks.append(peak)

    if output_single_peaks:
        return np.array(out), np.array(out_peaks)
    else:
        return np.array(out)

def sps_freeamp_fit(x, ped = 50, gain = 50, width_base = 5, width_scale = 1, ped_offs = 0, *ampls):

    out = np.zeros_like(x)

    for i, ampl in enumerate(ampls):
        mu = ped + gain * i
        if i>0:
            mu += ped_offs
            
        sigma = np.sqrt(width_base**2 +  width_scale**2 * i)

        peak = ampl * gauss(x, mu, sigma)
        
        out += peak

    return np.array(out)
    
def sps_freeamp(x, ped = 50, gain = 50, width_base = 5, width_scale = 1, ped_offs = 0, *ampls, output_single_peaks = False):
    print (len(x))
    out = np.zeros_like(x)
    out_peaks = []


    for i, ampl in enumerate(ampls):
        mu = ped + gain * i
        if i>0:
            mu += ped_offs
            
        sigma = np.sqrt(width_base**2 +  width_scale**2 * i)

        peak = ampl * gauss(x, mu, sigma)
        
        out += peak
        out_peaks.append(peak)
    ##trying to debug
    #print (len(out))
    #print (out_peaks)
    #print (len(out_peaks))
    if output_single_peaks:
        return np.array(out), np.array(out_peaks)
    else:
        return np.array(out)
    

def poisson_ampls(poisson_k):
    r = np.array([scipy.stats.poisson.pmf(n_pe, poisson_k) for n_pe in np.arange(0, int((3*poisson_k)+5), 1)])
    print(r)
    return r

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    

    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def CalculateSiPMGain(data,peakfinderWidth,directory,plotName,base,SiPM):
    content, bins, _ = plt.hist(data['data']['high_gain'],bins=np.max(data['data']['high_gain'])
                            ,range=(0,np.max(data['data']['high_gain']+1)),
         histtype='step', density  = True)
    centers = (bins[:-1] + bins[1:])/2
    yhat = savitzky_golay(content, 51, 3)
    print("done filtering the data")
    
    #plt.plot(centers,yhat, color='red')
    #plt.xlim(0,np.max(data['data']['high_gain']))
    
    peaks_in_interval = scipy.signal.find_peaks_cwt(yhat, widths=peakfinderWidth)
    peak_dist = np.diff(peaks_in_interval)
    print (np.median(peak_dist),"peakDist")
    print (peak_dist)
    
    print (np.median(data['data']['high_gain']))
    print (np.median(data['data']['high_gain'])/np.median(peak_dist))
    '''
    content, bins, _ = plt.hist(data['data']['high_gain'],bins=np.max(data['data']['high_gain'])
                            ,range=(0,np.max(data['data']['high_gain']+1)),
         histtype='step', density  = True)

    centers = (bins[:-1] + bins[1:])/2'''
    
    #print (peak_dist,"peak Dist")
    x = np.linspace(0,np.max(data['data']['high_gain']), len(content))
    r, cov = scipy.optimize.curve_fit(sps, centers, content, p0 = [0, np.median(peak_dist), base, base, np.median(data['data']['high_gain'])/np.median(peak_dist)])
    multigauss, peaks =  sps(x, *r, output_single_peaks=True)
    print (r[1],"gain for the poissonk fit")
    print ("Done with the first fir with same amplitudes ")
    #plt.show()
    plt.plot(x, multigauss, color='C2',label = f'Multi-Gauss Fit, gain = {r[1]:.2f} ADC/p.e.')
    for peak in peaks:
        plt.plot(x, peak, '--', color = 'C1')
        #print ("plotting individual peaks for poisson k fit")
    plt.plot(x, sps(x, *r),color='C3')
    print("trying to plot the SPS")
    plt.xlabel('Signal Amplitude in ADC')
    plt.ylabel('Normalized Counts')
    plt.title(SiPM+" @ 3.5 Vov")
    plt.legend(loc='best')
    #plt.legend(fontsize = 'x-large')
    #plt.figure()
    plt.savefig(directory + '/' + 'SPE_SPSwithoutSingleGaussians_'+plotName+'.png')
    plt.clf()
    print ("after saving the sps picture")

    #plt.close(fig)
    print ("skipping closing the sps picture")
    #plt.show()
    '''
    content, bins, _ = plt.hist(data['data']['high_gain'],bins=np.max(data['data']['high_gain'])
                            ,range=(0,np.max(data['data']['high_gain']+1)),
         histtype='step', density  = True)

    centers = (bins[:-1] + bins[1:])/2'''
    print ("second fit on the way")
    r2, cov2 = scipy.optimize.curve_fit(sps_freeamp_fit, centers, content, p0 = [*r[:-1], 0, *poisson_ampls(r[-1])])
    print (r2[1]," this the ADC/PE gain")

    multigauss1, peaks1 =  sps_freeamp(x, *r2, output_single_peaks=True)
    print ("After doing the free amp fit, almost done")
    print (multigauss1,"printing multigaus info")
    plt.plot(x, multigauss1, color='C2', label = f'Multi-Gauss Fit, gain = {r2[1]:.2f} ADC/p.e.', lw = 1)
    
    for peak in peaks1:
        plt.plot(x, peak, '--', color = 'C1')
        #print ("plotting individual free amp peaks")
    
    plt.xlabel('Signal Amplitude in ADC')
    plt.ylabel('Normalized Counts')
    plt.title(SiPM+" @ 3.5 Vov")
    plt.legend(loc='best')
    plt.savefig(directory + '/' + 'SPE_withoutSingleGaussians_'+plotName+'.png')
    plt.clf()
    #plt.close(fig)
    print ("plotting free amp fit")
    print (r2[1],"this is after plotting, printing gain")
    #plt.figure()
    plt.plot(centers,content)
    plt.plot(x, multigauss, color='C1', label = f'Multi-Gauss Fit, gain = {r2[1]:.2f} ADC/p.e.', lw = 3)
    plt.xlabel('Signal Amplitude in ADC')
    plt.ylabel('Normalized Counts')
    plt.title(SiPM+" @ 3.5 Vov")
    plt.legend(loc='best')
    plt.savefig(directory + '/' + 'SPE_withoutSingleGaussians_'+plotName+'.png')
    plt.clf()
    #plt.close(fig)
    #plt.show()
    print (r2[1],"this is after plotting pt.2, printing gain")
    plt.plot(x, multigauss, color='C1', label = f'Poisson K Multi-Gauss Fit, gain = {r[1]:.2f} ADC/p.e.', lw =1)
    plt.plot(x, multigauss1, color='C2', label = f'Free Amp Multi-Gauss Fit, gain = {r2[1]:.2f} ADC/p.e.', lw = 2)
    plt.xlabel('Signal Amplitude in ADC')
    plt.ylabel('Normalized Counts')
    plt.title(SiPM+" @ 3.5 Vov")
    plt.legend(loc='best')
    plt.savefig(directory + '/' + 'freeAmpvsPoissonK_'+plotName+'.png')
    plt.clf()
    #plt.close()
    #plt.show()
    percentageErr=(( sps(x, *r)-content)/content)*100
    percentageErr2=((multigauss1-content)/content)*100
    percentageErr3 =((sps(x, *r)-multigauss1)/multigauss1)*100
    print ("plotting free amp vs poisson k fit")
    print (r2[1],"this is before we return r2, the gain")
    return r2,np.median(percentageErr),np.median(percentageErr2),np.median(percentageErr3)
    
    ###

##Get the directory fron the arguments
parser = argparse.ArgumentParser()
parser.add_argument("inputDirectory",help="Input Directory to analyze")
parser.add_argument("typeOfData",help="Input type of data to analyze")
parser.add_argument("activeChannels",help="Number of channels in run")

#dataFile="/Users/irisponce/Documents/RHI/lfHCal/CorrectSPEfiles-0731/S14160-1315_57Gain_5p5Amplitude_40P0V/"
args=parser.parse_args()
activeChannels=int(args.activeChannels)
dataType=int(args.typeOfData)
dataFile=str(args.inputDirectory)

print ("Analyzing directory: ",dataFile)
data = parseData(args.inputDirectory,activeChannels)
print ("Data parsed, now proceeding to perfrom fits.")
#data=parseData("/Users/irisponce/Documents/RHI/lfHCal/SPEspectra_07312023/S4K33C0135L-9_57Gain_5p0Amplitude_30P5V/",1)
if dataFile.find("/") > dataFile.find("_"):##checking that we are in a subdirectory rather than directory
    SiPMName=dataFile[0:(dataFile.find("_"))]
else:
    SiPMName=dataFile[dataFile.find("/")+1:(dataFile.find("_"))]

print("SiPM ",SiPMName)
if dataType==1:
    print ("Making SPE fits")
    try:
        r2,percentageErr,percentageErr2,percentageErr3 = CalculateSiPMGain(data,20,dataFile,"try1",5,SiPMName)
        
        print (percentageErr,percentageErr2,percentageErr3)
    except:
        print ("Error ocurred during first fitting, will try again")
        pass

    ##I should check here that the values we get from the fitter actually make sense. Because we don't want to
    #rerun unless it's needed.

    if r2[1] <0:
        try:
            print ("Fitting again,we got a negative value for ADC/PE, take 2")
            r2,percentageErr,percentageErr2,percentageErr3 = CalculateSiPMGain(data,15,dataFile,"try2",5,SiPMName)
            print("try 2, ",percentageErr,percentageErr2,percentageErr3)
        except:
            
            pass
        
    if abs(percentageErr3)> 4:
        #try to refit not sure what should change LOL
        try:
            print ("Fitting again, the fitter worked but it wasn't a good fit trying again")
            r2,percentageErr,percentageErr2,percentageErr3 = CalculateSiPMGain(data,20,dataFile,"try3",15,SiPMName)
            
            print("try3, ",percentageErr,percentageErr2,percentageErr3)
        except:
            pass
    else:
        try:
            print ("Fitting again, the fitter worked but it wasn't a good fit trying again")
            r2,percentageErr,percentageErr2,percentageErr3 = CalculateSiPMGain(data,15,dataFile,"try3",0,SiPMName)
        
            print("try3, ",percentageErr,percentageErr2,percentageErr3)
        except:
            pass
        
elif dataType==2:
    print ("Analyzing cosmic data" )
    print ("STILL NEED TO DO THIS PART")
    content, bins, _ = plt.hist(data['data']['high_gain'],bins=np.max(data['data']['high_gain'])
                            ,range=(0,np.max(data['data']['high_gain']+1)),
                        histtype='step', density  = True)
    centers = (bins[:-1] + bins[1:])/2
    #plt.plot()

