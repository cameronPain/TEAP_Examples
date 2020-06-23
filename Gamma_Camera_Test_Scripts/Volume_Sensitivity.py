#!/usr/bin/env python3
#
#  Calculates the volume sensitivity of a system from a reconstructed SPECT image of a uniform cylindrical phantom. Paths are hard coded into the script relative to the base directory and dont need to be changed. The dose data is to be written into the series description of the dicom file in the format Dx:"activity:"CalTime" where Dx is either D1 or D2.
#
# cdp 20190724
#
import numpy as n
import matplotlib.pyplot as pyplot
from scipy.optimize import curve_fit as curve_fit
import pydicom
import os
from scipy.ndimage import convolve as convolve
import calendar
import time
import datetime
from   matplotlib.widgets import Slider


#Volume of the phantom used is 6.035 L
phantomVolume = 7.500




def binaryMask(Image, threshold):
    mask = n.ones(n.shape(Image))
    for i in range(len(Image)):
        for j in range(len(Image[0])):
            if Image[i,j] <= threshold:
                mask[i,j] = 0.0
            else:
                break
        for j in range(len(Image[0])):
            if Image[i,len(Image[0])-1-j] <= threshold:
                mask[i,j] = 0.0
            else:
                break
    return mask



def removeZeroPad(Image, threshold):
    zSum     = n.sum(Image,axis=0)
    for i in range(len(zSum)):
        if n.sum(zSum[i]) >= threshold:
            topIndex = i
            break
    for i in range(len(zSum)):
        if n.sum(zSum[len(zSum)-1-i]) >= threshold:
            bottomIndex = len(zSum)-1-i
            break

    tzSum = n.transpose(zSum)
    for i in range(len(tzSum)):
        if n.sum(tzSum[i]) >=threshold:
            leftIndex = i
            break
    for i in range(len(tzSum)):
        if n.sum(tzSum[len(tzSum)-1-i]) >= threshold:
            rightIndex = len(tzSum)-1-i
            break
    trimmedImage = Image[:,topIndex:bottomIndex,leftIndex:rightIndex]
    returnImage = []
    centreSliceThreshold = 0.75*n.sum(trimmedImage[int(len(trimmedImage)/2)])
    for slice in trimmedImage:
        if n.sum(slice)>=centreSliceThreshold:
            returnImage.append(slice)
    returnImage = n.array(returnImage)
    return returnImage



def checkArchiveForPreviousEntry(inputData, archive):
    inputDate    = inputData.split(',')[0]
    archiveData  = archive.read().split('\n')
    archiveDates = []
    for datString in archiveData:
        archiveDates.append(datString.split(',')[0])
    if inputDate not in archiveDates:
        return False
    else:
        if inputData in archiveData:
            print('Same data found in the log file.')
            return True
        else:
            return False



def cylindricalROI(Image, radius, mmPerPx):
    centreX, centreY = int(len(Image[0])/2), int(len(Image[0][0])/2)
    mask = n.ones(n.shape(Image))
    N    = n.shape(mask)[0]*n.shape(mask)[1]*n.shape(mask)[2]
    for k in range(len(mask)):
        for i in range(len(mask[k])):
            for j in range(len(mask[k][0])):
                distance = n.sqrt( ((centreX-j)**2) + ((centreY-i)**2) )*mmPerPx
                if distance > radius:
                    mask[k,i,j] = 0.0
                    N = N - 1
    maskedImage = n.multiply(mask, Image)
    return maskedImage, N





def main(srcfile, activity, calibrationTime, isotope, show_images = False):

    #gather QC data
    print('Gathering QC data.')
    print('Identifying correct isotope.')
    if    isotope == 'Tc99m':
        l = n.log(2)/(6.0067*60*60)
    elif isotope == 'I131':
        l = n.log(2)/(8.0252*24*60*60)
    elif isotope == 'Y90':
        l = n.log(2)/(64.053*60*60)
    elif isotope == 'Lu177':
        l = n.log(2)/(6.647*24*60*60)
    elif isotope == 'Ga67':
        l = n.log(2)/(3.2617*24*60*60)
    else:
        print('The specified istope is not configured in this node. Add the relevant information before processing the data.')
        return
    ds                          = pydicom.read_file(srcfile)
    pixelArray                  = ds.pixel_array

    if show_images:
        fig,(im_ax) = pyplot.subplots(1, 1, figsize=(21,11))
        im_ax.set_title('Input image')
        max_val     = n.amax(pixelArray)
        image       = im_ax.imshow(pixelArray[0], cmap = pyplot.cm.binary, vmin = 0.0, vmax = max_val)
        sliderAx    = pyplot.axes([0.1,0.1,0.18,0.02])
        sliceSlider = Slider(sliderAx, 'Slice', 0, len(pixelArray)-1, valstep = 1)
        def change_slice(val):
            new_val = int(val)
            image.set_data(pixelArray[new_val])
            fig.canvas.draw()
        sliceSlider.on_changed(change_slice)
        pyplot.show()

    mmPerPx                     = float(ds.PixelSpacing[0])
    centredImage                = removeZeroPad(pixelArray, 4*n.amax(n.sum(pixelArray,axis=0)))
    maskedImage, nonZeroPixels  = cylindricalROI(centredImage, 80.0, mmPerPx)

    if show_images:
        fig,(im_ax) = pyplot.subplots(1, 1, figsize=(21,11))
        im_ax.set_title('Masked image')
        max_val     = n.amax(maskedImage)
        image       = im_ax.imshow(maskedImage[0], cmap = pyplot.cm.binary, vmin = 0.0, vmax = max_val)
        sliderAx    = pyplot.axes([0.1,0.1,0.18,0.02])
        sliceSlider = Slider(sliderAx, 'Slice', 0, len(maskedImage)-1, valstep = 1)
        def change_slice(val):
            new_val = int(val)
            image.set_data(maskedImage[new_val])
            fig.canvas.draw()
        sliceSlider.on_changed(change_slice)
        pyplot.show()


#totalAcquisitionTime        = 2*int(ds.RotationInformationSequence[0].NumberOfFramesInRotation)*(float(ds.RotationInformationSequence[0].ActualFrameDuration)/1000.0)
    totalAcquisitionTime        = 2*int(48)*(float(ds.RotationInformationSequence[0].ActualFrameDuration)/1000.0)

    acquisitionTime             =  ds.AcquisitionTime
    calTimeStamp                = calibrationTime[0:2] + ':' + calibrationTime[2:4] + ':' + calibrationTime[4:6]
    startTimeStamp              = acquisitionTime[0:2] + ':' + acquisitionTime[2:4] + ':' + acquisitionTime[4:6]
    integralStart               = -1*int(calendar.timegm(time.strptime(calTimeStamp, '%H:%M:%S'))   - calendar.timegm(time.strptime(startTimeStamp, '%H:%M:%S')))
    integralEnd                 = integralStart + totalAcquisitionTime
    midPoint                    = integralStart + (totalAcquisitionTime/2.0)
    countsTotal                 = n.sum(pixelArray)
    CountRate                   = countsTotal/(integralEnd - integralStart)
    maskCounts                  = n.sum(maskedImage)
    dcMaskCountRate             = (maskCounts*l)/(n.exp(-l*integralStart) - n.exp(-l*integralEnd))
    activityConc                = activity/phantomVolume  #kBq/ml
    voiActivity                 = (activityConc/1000.0) * nonZeroPixels * ((mmPerPx/10.0)**3)
    VolumeSensitivity           = dcMaskCountRate/voiActivity

    print(len(pixelArray))
    print('')
    print(ds)
    print(totalAcquisitionTime)
    print('\n\n\n\n')
    print('Start Time: ' + str(startTimeStamp))
    print('integral start: ' + str(integralStart))
    print('integral end: ' + str(integralEnd))
    print('lineal pixel dim: ' + str(mmPerPx))
    print('Isotope:             ' + isotope)
    print('Activity (MBq):      ' + str(activity))
    print('Acquisition Time (s):' + str(totalAcquisitionTime))
    print('VOI counts: ' + str(maskCounts))
    print('Count Rate:  ' + str(dcMaskCountRate))
    print('VOI activity: ' + str(voiActivity))
    print('Activity Conc: ' + str(activityConc))
    print('Volume Sensitivity: ' + str(VolumeSensitivity) + ' cps/MBq')



import argparse


if __name__ == '__main__' :
  usage = 'Cameron Pain (cameron.pain@austin.org.au): Calculates the volume sensitivity of a system from a reconstructed SPECT image of a uniform cylindrical phantom. Paths are hard coded into the script relative to the base directory and dont need to be changed. The dose data is to be written into the series description of the dicom file in the format Dx:"activity:"CalTime" where Dx is either D1 or D2.'
  parser = argparse.ArgumentParser(description = usage)
  parser.add_argument('srcfolder', type = str,   help = 'The location of the data for analysis.')
  parser.add_argument('activity' , type = float, help = 'The residual subtracted activity in the field of view specified in units of MBq.')
  parser.add_argument('calTime'  , type = str,   help = 'The time at which the residual subtracted activity is calibrated to in the format HHMMSS. For example, 5:22:12 pm would be written 172212.')
  parser.add_argument('isotope'  , type = str,   help = 'The isotope tested. Accepted values are Tc99m, Lu177, Y90, Ga67 and I131.')
  parser.add_argument('--show_images', dest = 'show_images', default = False, action = 'store_true', help = 'Set this flag to view the analysed image.')
  args = parser.parse_args()
  main(args.srcfolder, args.activity, args.calTime, args.isotope, show_images = args.show_images)
#end ifls
