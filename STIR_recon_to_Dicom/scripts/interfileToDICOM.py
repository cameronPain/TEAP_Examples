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


def main(hdrFile, datFile, ofileName):
    #read in data
    hdrFileData = open(hdrFile, 'r').read()
    datFileData = open(datFile, 'rb').read()
    ds          = pydicom.read_file('src/template.dcm')
    data        = n.multiply(n.reshape(n.fromstring(datFileData, n.float32),[128,128,128]),10000.0)    
    
    intData     = data.astype(n.int16)
    stringInput = intData.tostring()

    #change the pixel data in the dicom file
    ds.PixelData = stringInput

    #change relevant dicom information
    ds.NumberOfFrames    = 128
    ds.NumberOfSlices    = 128
    ds.SeriesDescription       = hdrFile[:-3]
    ds.save_as(ofileName)










import argparse


if __name__ == '__main__' :
  usage = 'Written by Cameron Pain. Opens a dicom file.'
  parser = argparse.ArgumentParser(description = usage)
  parser.add_argument('hdrFile',  type = str, help = 'Path to the .hdr file.')
  parser.add_argument('datFile',  type = str, help = 'Path to the .dat file.')
  parser.add_argument('ofileName', type = str, default = 'NM', help = 'The file name of the output file. Exclude the .dcm extension.')
  args = parser.parse_args()
  main(args.hdrFile, args.datFile, args.ofileName)


  #end if
