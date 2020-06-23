#!/usr/bin/env python
#
# Measure the planar sensitivity from dicom file of raw data
#
# cdp 20171031
#


import pydicom, numpy as n, matplotlib.pyplot as pyplot, scipy.ndimage as ndi, scipy.fftpack as fftp
from scipy.misc import imresize
import os
import math

def main(srcfile1, cal_time, activity, isotope):

    ds1 = pydicom.read_file(srcfile1)

    acquisitionTime1   = float(ds1.ActualFrameDuration)/1000.0
    Image1  = ds1.pixel_array
    BGCounts = n.divide(n.sum(Image1[:10,:10]), 100)*n.product(n.shape(Image1))
    counts   = n.sum(Image1)
    counts1  = counts - BGCounts
    
    print(counts, BGCounts)
    print(ds1)
    print( '')
    print( 'Detector 1 Counts:')
    print( counts1)
    print( '')
    print( 'Detector 1 Count Rate:')
    print( counts1/acquisitionTime1)
    print( '')

    def time_difference_to_seconds(time1,time2):

        h1 = int(time1[0:2])*60*60
        m1 = int(time1[2:4])*60
        s1 = int(time1[4:6])
       
    
        h2 = int(time2[0:2])*60*60
        m2 = int(time2[2:4])*60
        s2 = int(time2[4:6])
    
        t1 = (h1 + m1 + s1)
        t2 = (h2 + m2 + s2)
        time_difference = (t2 - t1)
        
        print('Warning:')
        print('If the scan and calibration was performed on different days, the time difference calculation will not work.')
        print('')
        
        return time_difference
        
    startTime1 = ds1.SeriesTime
    
    print('')
    print('Time Difference: ')
    print(time_difference_to_seconds(cal_time, startTime1))
    print('')
    

    
    
    start_t1 = time_difference_to_seconds(cal_time, startTime1)
    end_t1 = start_t1 + (ds1.ActualFrameDuration/1000.)

    

    
    
    def Decay_Corrected_Activity(counts, startTime, endTime, HL):
        return (n.log(2)/HL)*counts*(1/(  n.exp((-n.log(2)*startTime)/HL) -  n.exp((-n.log(2)*endTime)/HL) ))

    if isotope == '99mTc':
        halfLife = 6.0067*60*60
    if isotope == '90Y':
        halfLife = 64.053*60*60
    if isotope == '131I':
        halfLife = 8.0252*24*60*60
    if isotope == '67Ga':
        halfLife = 3.2617*24*60*60
    if isotope == '177Lu':
        halfLife = 6.647*24*60*60




    measuredActivity1 = Decay_Corrected_Activity(counts1, start_t1, end_t1, halfLife)

    print ('corrected count rate: ' + str(measuredActivity1))


    Planar_Sensitivity_D1 = measuredActivity1/activity
    lowerBound1 = measuredActivity1/(activity + 0.05*activity)
    upperBound1 = measuredActivity1/(activity - 0.05*activity)
    

    

    print('')
    print('Planar Sensitivity: ' + str(Planar_Sensitivity_D1) + ' cps / MBq')
    print('Detector1 Upper Bound: ' + str(upperBound1))
    print('Detector1 Lower Bound: ' + str(lowerBound1))
    print('')
    print('Lower and upper bound taken for a +- 5% activity reading from dose calibrator.')
    print('')








import argparse

if __name__ == '__main__' :
  usage = 'cdp 20171031 cameron.pain@austin.org.au'
  parser = argparse.ArgumentParser(description = usage)

  parser.add_argument('dicom_file', type = str, help = 'NM dicom file with single detector data.')
  parser.add_argument('activity', type = float, help = ' Dose calibrator activity in MBq.')
  parser.add_argument('calibration_time', type = str, help = 'Time of activity calibration in format hhmmss. example: 5:47:22pm is written 174722')
  parser.add_argument('isotope', type = str, help = 'Isotope used in the measurement. \n 90Y, 177Lu, 67Ga, 99mTc, 131I')
  args = parser.parse_args()
  main(args.dicom_file , args.calibration_time, args.activity, args.isotope)
#end if



