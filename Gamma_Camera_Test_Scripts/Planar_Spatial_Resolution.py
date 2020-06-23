#!/usr/bin/env python3
#
# Calculates the integral and differential uniformities on the UFOV and CFOV of a flood image. The image post processing and analysis is done according to the NEMA NU1 protocol.
#
# cdp 20190827
#
import numpy as n
import matplotlib.pyplot as pyplot
from scipy.optimize import curve_fit as curve_fit
import pydicom
import os
from scipy.ndimage import convolve as convolve
from matplotlib2tikz import save as tikzsave

def Gaussian(E,E0,C,SBG,sigma):
    if type(E)!=n.ndarray and type(E)!=list:
        return C*n.exp(-((E-E0)**2)/(2*(sigma**2))) + SBG
    else:
        data = []
        for i in E:
            data.append(Gaussian(i, E0, C, SBG, sigma))
        return n.array(data)



def main(srcfile, orientation, singleDetector):
    ds         = pydicom.read_file(srcfile)
    pixelData  = ds.pixel_array
    if singleDetector:
        ImageD1    = pixelData
        ImageD2    = pixelData
        print('D1 and D2 data are equal.')
    else:
        ImageD1    = pixelData[0]
        ImageD2    = pixelData[1]
    linesD1    = []
    linesD2    = []

    if orientation == 'x':
        projectionImageD1 = n.sum(ImageD1, axis=0)
        projectionImageD2 = n.sum(ImageD2, axis=0)
    
    
    #pyplot.plot(n.arange(len(projectionImageD1)), projectionImageD1)
    #    pyplot.show()
    #   pyplot.plot(n.arange(len(projectionImageD2)), projectionImageD2)
    #   pyplot.show()

        maxD1 = n.amax(projectionImageD1)
        maxD2 = n.amax(projectionImageD2)
        linesIndicesD1 = []
        linesIndicesD2 = []
        
        for i in range(len(projectionImageD1)):
            if projectionImageD1[i]>=0.05*maxD1:
                d1start = i + 10
                print(i)
                break
        for i in range(len(projectionImageD1)):
            if projectionImageD1[len(projectionImageD1)-1-i] >= 0.5*maxD1:
                d1end   = len(projectionImageD1)-11-i
                break
        for i in range(len(projectionImageD2)):
            if projectionImageD2[i]>=0.05*maxD2:
                d2start = i + 10
                break
        for i in range(len(projectionImageD2)):
            if projectionImageD2[len(projectionImageD2)-1-i] >= 0.5*maxD2:
                d2end   = len(projectionImageD2)-11-i
                break

        for i in range(d1start, d1end):
            linesD1.append(ImageD1[:,i])
        for i in range(d2start, d2end):
            linesD2.append(ImageD2[:,i])



        maxPixLocD1 = []
        maxPixLocD2 = []
        for line in linesD1:
            maxima = n.amax(line)
            for i in range(len(line)):
                if line[i] == maxima:
                    maxPixLocD1.append(i)
                    break
        for line in linesD2:
            maxima = n.amax(line)
            for i in range(len(line)):
                if line[i] == maxima:
                    maxPixLocD2.append(i)
                    break


        linesD1 = n.array(linesD1)
        linesD2 = n.array(linesD2)
        print(n.shape(maxPixLocD1), n.shape(maxPixLocD2))
        print(n.shape(linesD1), n.shape(linesD2))
        trimLinesD1, trimlinesD2  = [] ,[]
        for i in range(len(maxPixLocD1)):
            #print( linesD1[i, maxPixLocD1[i]-30:maxPixLocD1[i]+30])
            trimLinesD1.append( linesD1[i, maxPixLocD1[i]-30:maxPixLocD1[i]+30])

        for j in range(len(maxPixLocD2)):
            #print( linesD2[i, maxPixLocD2[i]-30:maxPixLocD2[i]+30])
            trimlinesD2.append( linesD2[j, maxPixLocD2[j]-30:maxPixLocD2[j]+30])
        

    if orientation == 'y':
        projectionImageD1 = n.sum(ImageD1, axis=1)
        projectionImageD2 = n.sum(ImageD2, axis=1)
        pyplot.plot(projectionImageD1)
        pyplot.plot(projectionImageD2)
        pyplot.show()
        maxD1 = n.amax(projectionImageD1)
        maxD2 = n.amax(projectionImageD2)
        linesIndicesD1 = []
        linesIndicesD2 = []
       
        for i in range(len(projectionImageD1)):
            if projectionImageD1[i]>=0.5*maxD1:
                d1start = i + 10
                print(i)
                break
        for i in range(len(projectionImageD1)):
            if projectionImageD1[len(projectionImageD1)-1-i] >= 0.5*maxD1:
                d1end   = len(projectionImageD1)-11-i
                break
        for i in range(len(projectionImageD2)):
            if projectionImageD2[i]>=0.5*maxD2:
                d2start = i + 10
                break
        for i in range(len(projectionImageD2)):
            if projectionImageD2[len(projectionImageD2)-1-i] >= 0.5*maxD2:
                d2end   = len(projectionImageD2)-11-i
                break

        print('\n\n\n\n\n\n')
        print('D1 start and end.')
        print(d1start, d1end)
        print('\n\n\n\n\n\n')
        
        for i in range(d1start, d1end):
            linesD1.append(ImageD1[i,:])
        for i in range(d2start, d2end):
            linesD2.append(ImageD2[i,:])


        maxPixLocD1 = []
        maxPixLocD2 = []
        for line in linesD1:
            maxima = n.amax(line)
            for i in range(len(line)):
                if line[i] == maxima:
                    if i >= 200:
                        maxPixLocD1.append(i)
                        break
        for line in linesD2:
            maxima = n.amax(line)
            for i in range(len(line)):
                if line[i] == maxima:
                    maxPixLocD2.append(i)
                    break

        print(maxPixLocD1, maxPixLocD1)


        linesD1 = n.array(linesD1)
        linesD2 = n.array(linesD2)
        print(n.shape(maxPixLocD1), n.shape(maxPixLocD2))
        print(n.shape(linesD1), n.shape(linesD2))
        trimLinesD1, trimlinesD2  = [] ,[]
        for i in range(len(maxPixLocD1)):
            #print( linesD1[i, maxPixLocD1[i]-30:maxPixLocD1[i]+30])
            trimLinesD1.append( linesD1[i, maxPixLocD1[i]-30:maxPixLocD1[i]+30])

        for j in range(len(maxPixLocD2)):
            #print( linesD2[i, maxPixLocD2[i]-30:maxPixLocD2[i]+30])
            trimlinesD2.append( linesD2[j, maxPixLocD2[j]-30:maxPixLocD2[j]+30])







    centrePixel    = 30 #From trimming the lines around the maxima
    stdevEstimate  = 8.0 # estimating stdev of ~8 mm
    peakEstimate   = n.amax(ImageD1)
    dc             = 1.0
    pixels         = n.arange(len(trimlinesD2[0]))



    FWHMD1 = []
    FWTMD1 = []
    FWHMD2 = []
    FWTMD2 = []
    nlinesSkippedD1 = 0
    nlinesSkippedD2 = 0
    for i in range(len(trimLinesD1)):
        try: p1,c1 = curve_fit(Gaussian, pixels, trimLinesD1[i], p0=[centrePixel, peakEstimate, dc, stdevEstimate])
        except: nlinesSkippedD1 = nlinesSkippedD1 + 1
        print('D1 calculating...')
        fitData = n.arange(0,len(pixels),0.1)
        print(2*abs(p1[3])*n.sqrt(2*n.log(2))*float(ds.PixelSpacing[0]))
        covSum1 = n.sum(abs(c1))
        
        if i == 100:
        
            pyplot.scatter(pixels, trimLinesD1[i])
            pyplot.plot(fitData, Gaussian(fitData, *p1))
            pyplot.xlabel('Pixel Coordinate', fontsize=20)
            pyplot.ylabel('Counts', fontsize=20)
            pyplot.tick_params(axis='both', labelsize=20)
            tikzsave('LSF_LEHR.tex', figurewidth='16cm', figureheight='10cm')
            pyplot.show()
        if (covSum1 >1000 or p1[1]<1000.0):
            print(p1[3])
            print(n.sum(abs(c1)))
            pyplot.scatter(pixels, trimLinesD1[i])
            pyplot.plot(fitData, Gaussian(fitData, *p1))
            pyplot.xlabel('Pixel Coordinate', fontsize=20)
            pyplot.ylabel('Counts', fontsize=20)
            pyplot.tick_params(axis='both', labelsize=20)
            pyplot.show()
            nlinesSkippedD2 = nlinesSkippedD2 + 1
        if (covSum1 <1000 and p1[1] >= 1000.0):
            FWHMD1.append(abs(p1[3])*2*n.sqrt(2*n.log(2)))
            FWTMD1.append(abs(p1[3])*2*n.sqrt(2*n.log(10)))
    nlinesSkippedD1 = 0
    for i in range(len(trimlinesD2)):
        try: p2,c2 = curve_fit(Gaussian, pixels, trimlinesD2[i], p0=[centrePixel, peakEstimate/10.0, dc, stdevEstimate])
        except: nlinesSkippedD2 = nlinesSkippedD2 + 1
        print('D2 calculating...')
        fitData = n.arange(0,len(pixels),0.1)
        covSum2 = n.sum(abs(c2))
        print(2*abs(p2[3])*n.sqrt(2*n.log(2))*float(ds.PixelSpacing[0]))
        if (covSum2 > 1000 or p2[1]<1000.0):
            print(p2[3])
            print(n.sum(abs(c2)))
            pyplot.scatter(pixels, trimlinesD2[i])
            pyplot.plot(fitData, Gaussian(fitData, *p2))
            pyplot.show()
            nlinesSkippedD2 = nlinesSkippedD2 + 1

        if (covSum2 <1000and p2[1]>= 1000.0):
            FWHMD2.append(2*abs(p2[3])*n.sqrt(2*n.log(2)))
            FWTMD2.append(2*abs(p2[3])*n.sqrt(2*n.log(10)))
    print('Number of lines skipped (D1,D2): ' + str([nlinesSkippedD1, nlinesSkippedD2]))

    meanFWHMD1, stdevFWHMD1 = n.average(FWHMD1), n.std(FWHMD1)
    meanFWTMD1, stdevFWTMD1 = n.average(FWTMD1), n.std(FWTMD1)

    meanFWHMD2, stdevFWHMD2 = n.average(FWHMD2), n.std(FWHMD2)
    meanFWTMD2, stdevFWTMD2 = n.average(FWTMD2), n.std(FWTMD2)


    mmPerPixel = float(ds.PixelSpacing[0])


    print('Planar spatial resolution: ')
    print('     Detector 1:')
    print('         FWHM:' + str(mmPerPixel*meanFWHMD1) + ' +- ' + str(mmPerPixel * stdevFWHMD1) + ' mm')
    print('         FWTM:' + str(mmPerPixel*meanFWTMD1) + ' +- ' + str(mmPerPixel * stdevFWTMD1) + ' mm')
    print('     Detector 2:')
    print('         FWHM:' + str(mmPerPixel*meanFWHMD2) + ' +- ' + str(mmPerPixel * stdevFWHMD2) + ' mm')
    print('         FWTM:' + str(mmPerPixel*meanFWTMD2) + ' +- ' + str(mmPerPixel * stdevFWTMD2) + ' mm')




import argparse


if __name__ == '__main__' :
  usage = 'Written by Cameron Pain. Calculates the integral and differential uniformities on the UFOV and CFOV of a flood image. The image post processing and analysis is done according to the NEMA NU1 protocol.'
  parser = argparse.ArgumentParser(description = usage)
  parser.add_argument('dicom_file', type = str, help = 'Specify the path to the .dcm file containing the uniformity images.')
  parser.add_argument('--orientation', type = str, default = 'x', help = 'Set this either "x" or "y" based on the direction OF THE LINE SOURCE. x is horizontal and y vertical. Default is arbitrarily x.')
  parser.add_argument('--singleDetector', dest = 'singleDetector', default = False, action = 'store_true', help = 'Set this flag if you acquired data one detector at a time.')
  args = parser.parse_args()
  main(args.dicom_file, args.orientation, args.singleDetector)
#end ifls
