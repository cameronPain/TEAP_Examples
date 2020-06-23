#!/usr/bin/env python3

# Measure the energy resolution of a gamma camera using a .dcm file containing an image for each fine energy bin of the energy spectrum.
# cdp 20190827


import numpy as n
import matplotlib.pyplot as pyplot
import pydicom
import os
from scipy.optimize import curve_fit as curve_fit
import sys
from matplotlib2tikz import save as tikzsave

def Gaussian(E,E0,C,SBG,sigma):
    if type(E)!=n.ndarray and type(E)!=list:
        return C*n.exp(-((E-E0)**2)/(2*(sigma**2))) + SBG
    else:
        data = []
        for i in E:
            data.append(Gaussian(i, E0, C, SBG, sigma))
        return n.array(data)

def main(srcfile, isotope):
    ds        = pydicom.read_file(srcfile)
    pixelData = ds.pixel_array
    energyBin = []
    countsD1  = []
    countsD2  = []
    framesPerDetector = int(len(pixelData)/2.0)
    for i in range(len(ds.EnergyWindowInformationSequence)):
        lower = ds.EnergyWindowInformationSequence[i].EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
        upper = ds.EnergyWindowInformationSequence[i].EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
        e     = (lower + upper)/2.0
        energyBin.append(e)
    for i in range(framesPerDetector):
        iCountsD1 = n.sum(pixelData[2*i])
        iCountsD2 = n.sum(pixelData[2*i + 1])
        countsD1.append(iCountsD1)
        countsD2.append(iCountsD2)

    #end for
    energyBin = n.array(energyBin)
    countsD1  = n.array(countsD1)
    countsD2  = n.array(countsD2)
    print('Doing least squares fit to a Gaussian...')
    if isotope == 'Co57':
        photopeak = 122.0
        print('Searching for photopeak at ' + str(photopeak) + ' keV.')
    elif isotope == 'I131':
        photopeak = 365.0
        print('Searching for photopeak at ' + str(photopeak) + ' keV.')
    elif isotope == 'Tc99m':
        photopeak = 140.5
        print('Searching for photopeak at ' + str(photopeak) + ' keV.')
    elif isotope == 'Ga67_93':
        photopeak = 93.0
        print('Searching for photopeak at ' + str(photopeak) + ' keV.')
    elif isotope == 'Ga67_300':
        photopeak = 300
        print('Searching forphotopeak at '  + str(photopeak) + ' keV.')
    elif isotope == 'In111':
        photopeak = 245.0
        print('Searching forphotopeak at '  + str(photopeak) + ' keV.')
    else:
        print('The isotope you chose is not programmed in. Check the code. Defaulting to Tc99m')
        photopeak = 140.5

    if isotope == 'Co57':
        countsD1 = []
        for image in pixelData:
            countsD1.append(n.sum(image))
        countsD1 = n.array(countsD1)
        countsD2 = countsD1

    pyplot.scatter(energyBin, countsD1)
    pyplot.scatter(energyBin, countsD2)
    pyplot.show()


    peakEstimate = n.sum(pixelData[framesPerDetector])
    SBGestimate  = 0.04*peakEstimate
    paramsD1, covD1 = curve_fit(Gaussian, energyBin, countsD1, p0=[photopeak, peakEstimate, SBGestimate, 0.05*photopeak])
    paramsD2, covD2 = curve_fit(Gaussian, energyBin, countsD2, p0=[photopeak, peakEstimate, SBGestimate, 0.05*photopeak])
    print('Fit parameters calculated.')
    print('Check the quality of the fit...')
    Erange = n.arange(0.75*photopeak, 1.25*photopeak, 0.005*photopeak)
    fitD1  = Gaussian(Erange, *paramsD1)
    fitD2  = Gaussian(Erange, *paramsD2)
    pyplot.scatter(energyBin, countsD1, color='b', label='Detector 1')
    pyplot.scatter(energyBin, countsD2, color='r', label='Detector 2')
    pyplot.plot(Erange, fitD1, '--', color='b')
    pyplot.plot(Erange, fitD2, '--', color='r')
    pyplot.xlabel('Energy (keV)', fontsize=25)
    pyplot.ylabel('Counts', fontsize=25)
    pyplot.tick_params(axis='both',labelsize=25)
    pyplot.legend()
    tikzsave('EnergyResolutionTc99m.tex', figureheight='10cm', figurewidth='16cm')
    pyplot.show()

    FWHMD1, FWTMD1 = 2*paramsD1[3]*n.sqrt(2*n.log(2)), 2*paramsD1[3]*n.sqrt(2*n.log(10))
    FWHMD2, FWTMD2 = 2*paramsD2[3]*n.sqrt(2*n.log(2)), 2*paramsD2[3]*n.sqrt(2*n.log(10))


    print('')
    print('Energy resolutions: ')
    print('     Detector 1: ')
    print('         FWHM = ' + str(abs(FWHMD1)) + ', ' + str(100*abs((FWHMD1/photopeak))) + '%')
    print('         FWTM = ' + str(abs(FWTMD1)) + ', ' + str(100*abs((FWTMD1/photopeak))) + '%')

    print('     Detector 2: ')
    print('         FWHM = ' + str(abs(FWHMD2)) + ', ' + str(100*abs((FWHMD2/photopeak))) + '%')
    print('         FWTM = ' + str(abs(FWTMD2)) + ', ' + str(100*abs((FWTMD2/photopeak))) + '%')
    print('Photopeak from fit: ')
    print('     Detector 1: ')
    print('         photopeak = ' + str(abs(paramsD1[0]))+ ' keV')
    print('     Detector 2: ')
    print('         photopeak = ' + str(abs(paramsD2[0]))+ ' keV')
    print('')





import argparse

if __name__ == '__main__' :
  usage = 'cdp 20190827 cameron.pain@austin.org.au: Return the energy resolution of a scanner using a single .dcm file containing an image for each sampling energy window. Requires that an energy session be made with unique energy windows sampling the photopeak into bins (images).'
  parser = argparse.ArgumentParser(description = usage)
  parser.add_argument('srcfile', type = str, help = 'The path to the .dcm file containing the energy spectrum data. It should be a single dicom file containing images each acquired at the +- 0.5%% energy bins defined in the energy window protocol.')
  parser.add_argument('isotope', type = str, default = 'Tc99m', help = 'The isotope tested: I131, Co57 or Tc99m.')
  args = parser.parse_args()
  main(args.srcfile, args.isotope)
#end if



