#!/usr/bin/env python3
#
# Calculates the SPECT CT alignment from a three point tomographic acquisition.
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
from DicomFolderRead import DicomFolderRead as DFR
import sys
import time

def gaussian(x,x0,A,sigma,C):
    if type(x)!=n.ndarray and type(x)!=list:
        return A*n.exp(-((x-x0)**2)/(2*(sigma**2))) + C
    else:
        data = []
        for i in x:
            data.append(gaussian(i,x0,A,sigma,C))
        return n.array(data)




def thresholdVolume(Image, threshold):
    dim  = n.shape(Image)
    data = n.zeros(dim)
    for i in range(len(Image)):
        sys.stdout.write('\rCreating CT threshold Image %d' %i)
        sys.stdout.flush()
        for j in range(len(Image[0])):
            for k in range(len(Image[0][0])):
                if Image[i][j][k] >= threshold:
                   data[i][j][k] =  Image[i][j][k]
    returnImage = n.reshape(data, dim)
    return returnImage



def maximaLocatorCT(Image, N, debug): # return an (x,y,z) coordinate of the maxima
    projAxial = n.sum(Image, axis = 0)
    maxima    = []
    for iter in range(N):
        max   = n.amax(projAxial)
        for i in range(len(projAxial)):
            for j in range(len(projAxial[0])):
                if projAxial[i][j] == max:
                    maxima.append([i,j])
                    if debug:
                        pyplot.imshow(projAxial)
                        pyplot.title('axial projection')
                        pyplot.show()
                    projAxial[i-10:i+10,j-10:j+10] = n.zeros([20,20])
                    break
                    break
    projCoronal = n.sum(Image,axis = 1)
    maxCor      = []
    for iter in range(N):
        max   = n.amax(projCoronal)
        for i in range(len(projCoronal)):
            for j in range(len(projCoronal[0])):
                if projCoronal[i][j] == max:
                    maxCor.append([i,j])
                    if debug:
                        pyplot.imshow(projCoronal)
                        pyplot.title('coronal projection')
                        pyplot.show()
                    projCoronal[i-10:i+10,j-10:j+10] = n.zeros([20,20])
                    break
                    break
    return maxima, maxCor


def centroidData(Image):
    data = []
    print('Finding centroid data.')
    for i in range(len(Image)):
        for j in range(len(Image[0])):
            for k in range(len(Image[0][0])):
                if Image[i][j][k] > 0.0:
                    print('Centroid: ')
                    data.append([[i,j,k],Image[i,j,k]])
    return data


def main(SPECTdata, CTdata, debug):
    dsSPECT    = pydicom.read_file(SPECTdata)
    dsCT       = []
    CTfiles    = DFR(CTdata)
    CTarray    = []
    for file in CTfiles:
        dsCT.append(pydicom.read_file(file))
        CTarray.append(pydicom.read_file(file).pixel_array)
    SPECTarray = dsSPECT.pixel_array
    CTarray    = n.array(CTarray)

    SPECTcoorAxial, SPECTcoorCoronal = maximaLocatorCT(SPECTarray, 3, debug)
    SPECTVOIs = []
    xyzCoordsSPECT = [[SPECTcoorAxial[1][1],SPECTcoorAxial[1][0],SPECTcoorCoronal[1][0]], [SPECTcoorAxial[2][1],SPECTcoorAxial[2][0],SPECTcoorCoronal[2][0]],[SPECTcoorAxial[0][1],SPECTcoorAxial[0][0],SPECTcoorCoronal[0][0]]]
    for coord in xyzCoordsSPECT:
        SPECTVOIs.append( SPECTarray[coord[2]-10:coord[2]+10, coord[1]-10:coord[1]+10, coord[0]-10:coord[0]+10]  )

    if debug:
        for VOI in SPECTVOIs:
            for slice in VOI:
                pyplot.imshow(slice, cmap = pyplot.cm.hot, vmin = 0.0, vmax = n.amax(VOI))
                pyplot.title('Localised point source VOIs')
                pyplot.show()

    SPECTcentreOfMass = []
    for VOI in SPECTVOIs:
        COM  = [0,0,0]
        norm = 1/n.sum(VOI)
        for i in range(len(VOI)):
            for j in range(len(VOI[0])):
                for k in range(len(VOI[0][0])):
                    COM = n.add(COM, n.multiply(VOI[i][j][k], [i,j,k])  )
        normCOM = n.multiply(norm, COM)
        SPECTcentreOfMass.append(normCOM)
    reshapedSPECTcoord = []
    for coord in xyzCoordsSPECT:
        reshapedSPECTcoord.append([coord[2],coord[1],coord[0]-1])
    originalSPECTcoordCOM = []
    for i in range(3):
        originalSPECTcoordCOM.append(n.add(n.add( reshapedSPECTcoord[i] , SPECTcentreOfMass[i]), -10))
    print('')
    print('C.O.M VOIs:')
    for i in originalSPECTcoordCOM:
        print('     ' + str(i) )
    print('')
    TrimmedImage   = CTarray[120-43:300-43,128:3*128,128:3*128]
    thresholdImage = thresholdVolume(TrimmedImage, 200)
    maxima, maxCor         = maximaLocatorCT(thresholdImage, 3, debug)
    xyzCoords = [[maxima[1][1],maxima[1][0],maxCor[1][0]], [maxima[2][1],maxima[2][0],maxCor[2][0]],[maxima[0][1],maxima[0][0],maxCor[0][0]]]
    VOIs = []
    for coord in xyzCoords:
        VOIs.append( thresholdImage[coord[2]-10:coord[2]+10, coord[1]-10:coord[1]+10, coord[0]-10:coord[0]+10]  )

    if debug:
        for VOI in VOIs:
            for slice in VOI:
                pyplot.imshow(slice, cmap = pyplot.cm.hot, vmin = 0.0, vmax = n.amax(VOI))
                pyplot.title('Localised point source VOIs')
                pyplot.show()

    centreOfMass = []
    for VOI in VOIs:
        COM  = [0,0,0]
        norm = 1/n.sum(VOI)
        for i in range(len(VOI)):
            for j in range(len(VOI[0])):
                for k in range(len(VOI[0][0])):
                    COM = n.add(COM, n.multiply(VOI[i][j][k], [i,j,k])  )
        normCOM = n.multiply(norm, COM)
        centreOfMass.append(normCOM)
    reshapedCoord = []
    for coord in xyzCoords:
        reshapedCoord.append([coord[2],coord[1],coord[0]])
    originalTrim     = [110-43, 118, 118]
    originalCoordCOM = []
    for i in range(len(centreOfMass)):
        originalCoordCOM.append(    n.add(  n.add(originalTrim, reshapedCoord[i]), centreOfMass[i] )    )
    print('')
    print('C.O.M VOIs:')
    for i in originalCoordCOM:
        print('     ' + str(i) )
    print('')


    CTpixelDimensions    = [0.625, float(dsCT[0].PixelSpacing[0]), float(dsCT[0].PixelSpacing[1])]
    SPECTpixelDimensions = [float(dsSPECT.SliceThickness), float(dsSPECT.PixelSpacing[0]), float(dsSPECT.PixelSpacing[1])]


    print('')
    print(CTpixelDimensions)
    print(SPECTpixelDimensions)
    print('')


    SPECT_centreOffset   = [128,127,128]
    CT_centreOffset      = [163.9,256,256]

    originalSPECTcoordCOM = n.subtract(originalSPECTcoordCOM, SPECT_centreOffset)
    originalCoordCOM    = n.subtract(originalCoordCOM,    CT_centreOffset)


    print('')
    print(' CT Pixel dimensions:')
    print('    x: ' + str(CTpixelDimensions[2]) )
    print('    y: ' + str(CTpixelDimensions[1]) )
    print('    z: ' + str(CTpixelDimensions[0]) )
    print('')
    print(' SPECT Pixel dimensions:')
    print('    x: ' + str(SPECTpixelDimensions[2]) )
    print('    y: ' + str(SPECTpixelDimensions[1]) )
    print('    z: ' + str(SPECTpixelDimensions[0]) )
    print('')

    SPECT_COMmm = n.multiply(SPECTpixelDimensions, originalSPECTcoordCOM)
    CT_COMmm    = n.multiply(CTpixelDimensions,    originalCoordCOM)

    print('')
    print('C.O.M SPECT VOIs (mm):')
    for i in SPECT_COMmm:
        print('     ' + str(i) )
    print('')
    print('')
    print('C.O.M CT VOIs (mm):')
    for i in CT_COMmm:
        print('     ' + str(i) )
    print('')

    maxDifference = []
    xyzDifference = []
    for i in range(3):
        maxDifference.append(   n.sqrt( ((CT_COMmm[i][0] - SPECT_COMmm[i][0])**2) + ((CT_COMmm[i][1] - SPECT_COMmm[i][1])**2) + ((CT_COMmm[i][2] - SPECT_COMmm[i][2])**2)     )    )
        xyzDifference.append(      [n.sqrt( ((CT_COMmm[i][0] - SPECT_COMmm[i][0])**2)), n.sqrt( ((CT_COMmm[i][1] - SPECT_COMmm[i][1])**2)), n.sqrt( ((CT_COMmm[i][2] - SPECT_COMmm[i][2])**2))]         )




    print('')
    print('Difference in CT and SPECT locations (mm):')
    for i in range(len(maxDifference)):
        print('Point ' + str(i) + ':')
        print('max difference : ' + str(maxDifference[i]))
        print('             x : ' + str(xyzDifference[i][2]))
        print('             y : ' + str(xyzDifference[i][1]))
        print('             z : ' + str(xyzDifference[i][0]))
    print('')








import argparse

if __name__ == '__main__' :
  usage = 'Written by Cameron Pain. Calculates the integral and differential uniformities on the UFOV and CFOV of a flood image. The image post processing and analysis is done according to the NEMA NU1 protocol.'
  parser = argparse.ArgumentParser(description = usage)
  parser.add_argument('SPECTdata', type = str, help = 'Path to the SPECT dataset.')
  parser.add_argument('CTdataFolder',    type = str, help = 'Path to the CT dataset.')
  parser.add_argument('--debug',    dest = 'debug', default = False, action = 'store_true', help = 'Set this flag to show images etc for analysis.')
  args = parser.parse_args()
  main(args.SPECTdata, args.CTdataFolder, args.debug)
#end ifls
