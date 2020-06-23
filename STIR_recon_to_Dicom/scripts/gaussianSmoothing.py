#!/usr/bin/env python3
#
# Open a dicom file with pydicom.
#
# cdp 20180703
#
import numpy as n
import matplotlib.pyplot as pyplot
from scipy.optimize import curve_fit as curve_fit
import pydicom
import os
import calendar
import time
import datetime
import matplotlib.dates as mdates
from scipy.ndimage import convolve as convolve
from scipy.ndimage import rotate as rotate
import scipy.ndimage as ndi

def main(dcm1, ofileName):
    print('Applying smoothing...')
    #read in data
    ds1        = pydicom.read_file(dcm1)
    #cut and paste pixel data
    pixelArray = ds1.pixel_array
    pixelSize  = float(ds1.PixelSpacing[0])
    pixelArray = pixelArray[:,:,::-1] # mirror the array about the y axis. STIR reconstructed it this way.
    
    def threeDimensionalGaussian(x,y,z, x0,y0,z0, s):
        return 1000* n.exp(-((x-x0)**2)/(2*(s**2))) * n.exp(-((y-y0)**2)/(2*(s**2))) * n.exp(-((z-z0)**2)/(2*(s**2)))
    
    #Use an odd number so there is a centre pixel.
    xdim = 3
    ydim = 3
    zdim = 3
    nPxIn10mm  = 5.0/pixelSize
    x0   = int(xdim/2)
    y0   = int(ydim/2)
    z0   = int(zdim/2)
    
    gaussianKernel = n.ones([xdim,ydim,zdim]).astype(n.float32)
    for i in range(len(gaussianKernel)):
        for j in range(len(gaussianKernel[0])):
            for k in range(len(gaussianKernel[0][0])):
                gaussianKernel[i,j,k] = threeDimensionalGaussian(i, j, k, x0, y0, z0, nPxIn10mm)


    gaussianKernel = n.multiply(gaussianKernel, 1/n.sum(gaussianKernel))
    smoothedPixelArray = ndi.convolve(pixelArray, gaussianKernel)
    smoothedPixelArray = n.multiply(smoothedPixelArray, (200/n.amax(smoothedPixelArray))).astype(n.int16)
    ds1.PixelData = smoothedPixelArray.tostring()
    ds1.save_as(ofileName)


import argparse


if __name__ == '__main__' :
  usage = 'Written by Cameron Pain. Opens a dicom file.'
  parser = argparse.ArgumentParser(description = usage)
  parser.add_argument('dcm1', type = str, help = 'The source dicom file.')
  parser.add_argument('ofileName', type = str, default = 'ofile', help = 'Specify the name of the output file.')
  args = parser.parse_args()
  main(args.dcm1, args.ofileName)


  #end if
