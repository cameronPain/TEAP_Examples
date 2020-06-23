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

def main(dcmFile, changMask, ofileName):
    print('Applying Chang correction...')
    #read in data
    ds1 = pydicom.read_file(dcmFile)
    ds2 = pydicom.read_file(changMask)
    rescaleSlope = float(ds2.RescaleSlope)
    #cut and paste pixel data
    pixelArray     = ds1.pixel_array
    changArray     = n.reshape(n.fromstring(ds2.PixelData, n.int16),n.shape(pixelArray))
    changArray     = n.multiply(changArray, rescaleSlope/1000.0)
    correctedImage = n.multiply(pixelArray,changArray)
    byteData       = correctedImage.astype(n.int16).tostring()
    ds1.PixelData = byteData
    ds1.SeriesDescription = ds1.SeriesDescription + '_changAC'
    ds1.save_as(ofileName + '.dcm')

#for slice in data:
#       pyplot.imshow(slice, cmap = pyplot.cm.binary, vmin= n.amin(data), vmax = n.amax(data))
#        pyplot.show()










import argparse


if __name__ == '__main__' :
  usage = 'Written by Cameron Pain. Opens a dicom file.'
  parser = argparse.ArgumentParser(description = usage)
  parser.add_argument('dcmFile', type = str, help = 'The source dicom file.')
  parser.add_argument('changFile', type = str, help = 'The paste dicom file.')
  parser.add_argument('ofileName', type = str, default = 'ofile', help = 'Specify the name of the output file.')
  args = parser.parse_args()
  main(args.dcmFile, args.changFile, args.ofileName)


  #end if
