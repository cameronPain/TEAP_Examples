#!/usr/bin/env python3
#
#
# Calculates the spatial resolution of the system from the PSF of the three point sources.
#
# Transaxial, Coronal and Sagital planes.
#
# Input data is SPECT data gathered using the "NEMA NU1 Performance Measurements of Gamma Cameras"
# System Alignment procedure.
#
#
# cdp  20170220

import pydicom, numpy as n, matplotlib.pyplot as pyplot
import matplotlib.animation as animation
import os
from scipy.optimize import curve_fit

def find_n_maxima(A, m):
    a = 0
    B = []
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i,j] >= n.amax(A)/3:
 
               a = a + 1
               B.append([i,j])
               A[i-6:i+6, j-6:j+6] = 0
               if a == m:
                  return B



def main(srcfile):
    dsTransaxial = pydicom.read_file(srcfile)
    print(dsTransaxial)

    #Transaxial
    zIntegratedImage = n.sum(dsTransaxial.pixel_array, axis = 0)
    pyplot.imshow(zIntegratedImage, cmap = pyplot.cm.hot, vmin = 0.0, vmax = n.amax(zIntegratedImage))
    pyplot.show()
    max_coordinates = find_n_maxima(n.sum(dsTransaxial.pixel_array,axis = 0),3)
    ROI = []
    for i in range(len(max_coordinates)):
        ROI.append(zIntegratedImage[max_coordinates[i][0]-10:max_coordinates[i][0]+10 ,max_coordinates[i][1]-10:max_coordinates[i][1]+10])
    Profiles = []
    for i in ROI:
        Profiles.append([n.sum(i,axis = 0), n.sum(i,axis=1)])

    def Gaussian_fit(x, A, x0, sigma):
        return A * n.exp(-(((x-x0)**2)/(2*(sigma**2))))
    
    def Gaussian(x, A, x0, sigma):
        p = []
        for i in x:
            p.append(     A*n.exp(-(((i-x0)**2)/(2*(sigma**2))))          )
        return p

    Gauss_Params = []

    for i in Profiles:
        y,d2 =curve_fit(Gaussian_fit,range(len(i[1])),i[1], maxfev=1000, p0 = [100, 30, 20] )
        x,d1 = curve_fit(Gaussian_fit,range(len(i[0])),i[0], maxfev=1000, p0 =[100, 30, 20] )
        Gauss_Params.append([x,y])

    FWHM = []
    FWTM = []
    for i in Gauss_Params:
        FWHM.append([i[0][2]*n.sqrt(2*n.log(2)), i[1][2]*n.sqrt(2*n.log(2))])
        FWTM.append([i[0][2]*n.sqrt(2*n.log(10)), i[1][2]*n.sqrt(2*n.log(10))])
    for i in range(len(Gauss_Params)):
        for j in range(len(Gauss_Params[0])):
            pyplot.scatter(n.arange(len(Profiles[i][j])), Profiles[i][j])
            pyplot.plot(n.arange(0,len(Profiles[0][0]),0.01), Gaussian(n.arange(0,len(Profiles[0][0]),0.01), Gauss_Params[i][j][0],Gauss_Params[i][j][1],Gauss_Params[i][j][2] ), color = 'k')
            pyplot.show()


    pixel_size = dsTransaxial.PixelSpacing[0]
    print( '')
    print( '')
    print( 'Transaxial:')
    print( 'Point Source 1: ')
    print( ' FWHM x: ' + str(abs(FWHM[0][0])*pixel_size) + ' mm')
    print( ' FWHM y: ' + str(abs(FWHM[0][1])*pixel_size) + ' mm')
    print( ' FWTM x: ' + str(abs(FWTM[0][0])*pixel_size) + ' mm')
    print( ' FWTM y: ' + str(abs(FWTM[0][1])*pixel_size) + ' mm')
    print( '')
    print( 'Point Source 2: ')
    print( ' FWHM x: ' + str(abs(FWHM[1][0])*pixel_size) + ' mm')
    print( ' FWHM y: ' + str(abs(FWHM[1][1])*pixel_size) + ' mm')
    print( ' FWTM x: ' + str(abs(FWTM[1][0])*pixel_size) + ' mm')
    print( ' FWTM y: ' + str(abs(FWTM[1][1])*pixel_size) + ' mm')
    print( '')
    print( 'Point Source 3: ')
    print( ' FWHM x: ' + str(abs(FWHM[2][0])*pixel_size) + ' mm')
    print( ' FWHM y: ' + str(abs(FWHM[2][1])*pixel_size) + ' mm')
    print( ' FWTM x: ' + str(abs(FWTM[2][0])*pixel_size) + ' mm')
    print( ' FWTM y: ' + str(abs(FWTM[2][1])*pixel_size) + ' mm')
    print( '')

    #Coronal
    yIntegratedImage = n.sum(dsTransaxial.pixel_array, axis = 1)
    max_coordinates = find_n_maxima(n.sum(dsTransaxial.pixel_array,axis = 1),3)

    pyplot.imshow(yIntegratedImage, cmap = pyplot.cm.hot, vmin = 0.0, vmax = n.amax(yIntegratedImage))
    pyplot.show()


    
    ROI = []
    
    for i in range(len(max_coordinates)):
        ROI.append(yIntegratedImage[max_coordinates[i][0]-10:max_coordinates[i][0]+10 ,max_coordinates[i][1]-10:max_coordinates[i][1]+10])



    Profiles = []

    for i in ROI:
        Profiles.append([n.sum(i,axis = 0), n.sum(i,axis=1)])

    Gauss_Params = []

    for i in Profiles:
        
        y,d2 =curve_fit(Gaussian_fit,range(len(i[1])),i[1], maxfev=1000, p0 = [100, 30, 20] )
        x,d1 = curve_fit(Gaussian_fit,range(len(i[0])),i[0], maxfev=1000, p0 =[100, 30, 20] )
        Gauss_Params.append([x,y])




    FWHM = []
    FWTM = []
    
    for i in Gauss_Params:
        FWHM.append([i[0][2]*n.sqrt(2*n.log(2)), i[1][2]*n.sqrt(2*n.log(2))])
        FWTM.append([i[0][2]*n.sqrt(2*n.log(10)), i[1][2]*n.sqrt(2*n.log(10))])


    for i in range(len(Gauss_Params)):
        for j in range(len(Gauss_Params[0])):
            pyplot.scatter(n.arange(len(Profiles[i][j])),Profiles[i][j])
            pyplot.plot(n.arange(0,len(Profiles[0][0]),0.01), Gaussian(n.arange(0,len(Profiles[0][0]),0.01), Gauss_Params[i][j][0],Gauss_Params[i][j][1],Gauss_Params[i][j][2] ), color = 'k')
            pyplot.show()


    pixel_size = dsTransaxial.PixelSpacing[0]
    print( '')
    print( '')
    print( 'Coronal:')
    print( 'Point Source 1: ')
    print( ' FWHM x: ' + str(abs(FWHM[0][0])*pixel_size) + ' mm')
    print( ' FWHM y: ' + str(abs(FWHM[0][1])*pixel_size) + ' mm')
    print( ' FWTM x: ' + str(abs(FWTM[0][0])*pixel_size) + ' mm')
    print( ' FWTM y: ' + str(abs(FWTM[0][1])*pixel_size) + ' mm')
    print( '')
    print( 'Point Source 2: ')
    print( ' FWHM x: ' + str(abs(FWHM[1][0])*pixel_size) + ' mm')
    print( ' FWHM y: ' + str(abs(FWHM[1][1])*pixel_size) + ' mm')
    print( ' FWTM x: ' + str(abs(FWTM[1][0])*pixel_size) + ' mm')
    print( ' FWTM y: ' + str(abs(FWTM[1][1])*pixel_size) + ' mm')
    print( '')
    print( 'Point Source 3: ')
    print( ' FWHM x: ' + str(abs(FWHM[2][0])*pixel_size) + ' mm')
    print( ' FWHM y: ' + str(abs(FWHM[2][1])*pixel_size) + ' mm')
    print( ' FWTM x: ' + str(abs(FWTM[2][0])*pixel_size) + ' mm')
    print( ' FWTM y: ' + str(abs(FWTM[2][1])*pixel_size) + ' mm')
    print( '')

    #Sagital
    xIntegratedImage = n.sum(dsTransaxial.pixel_array, axis = 2)
    pyplot.imshow(xIntegratedImage, cmap = pyplot.cm.hot, vmin = 0.0, vmax = n.amax(xIntegratedImage))
    pyplot.show()
    
    max_coordinates = find_n_maxima(n.sum(dsTransaxial.pixel_array,axis = 2),3)
    
    
    ROI = []
    
    for i in range(len(max_coordinates)):
        ROI.append(xIntegratedImage[max_coordinates[i][0]-10:max_coordinates[i][0]+10 ,max_coordinates[i][1]-10:max_coordinates[i][1]+10])



    Profiles = []

    for i in ROI:
        Profiles.append([n.sum(i,axis = 0), n.sum(i,axis=1)])
    
    
    def Gaussian_fit(x, A, x0, sigma):
        return A * n.exp(-(((x-x0)**2)/(2*(sigma**2))))
    
    def Gaussian(x, A, x0, sigma):
        p = []
        for i in x:
            p.append(     A*n.exp(-(((i-x0)**2)/(2*(sigma**2))))          )
        return p
    
    Gauss_Params = []

    for i in Profiles:
        
        y,d2 =curve_fit(Gaussian_fit,range(len(i[1])),i[1], maxfev=1000, p0 = [100, 30, 20] )
        x,d1 = curve_fit(Gaussian_fit,range(len(i[0])),i[0], maxfev=1000, p0 =[100, 30, 20] )
        Gauss_Params.append([x,y])

    
    FWHM = []
    FWTM = []
    
    for i in Gauss_Params:
        FWHM.append([i[0][2]*n.sqrt(2*n.log(2)), i[1][2]*n.sqrt(2*n.log(2))])
        FWTM.append([i[0][2]*n.sqrt(2*n.log(10)), i[1][2]*n.sqrt(2*n.log(10))])
    
    
    for i in range(len(Gauss_Params)):
        for j in range(len(Gauss_Params[0])):
            pyplot.scatter(n.arange(len(Profiles[i][j])),Profiles[i][j])
            pyplot.plot(n.arange(0,len(Profiles[0][0]), 0.01), Gaussian(n.arange(0,len(Profiles[0][0]),0.01), Gauss_Params[i][j][0],Gauss_Params[i][j][1],Gauss_Params[i][j][2] ), color = 'k')
            pyplot.show()


    pixel_size = dsTransaxial.PixelSpacing[0]

    print('')
    print( '')
    print( 'Sagital:')
    print( 'Point Source 1: ')
    print( ' FWHM x: ' + str(abs(FWHM[0][0])*pixel_size) + ' mm')
    print( ' FWHM y: ' + str(abs(FWHM[0][1])*pixel_size) + ' mm')
    print( ' FWTM x: ' + str(abs(FWTM[0][0])*pixel_size) + ' mm')
    print( ' FWTM y: ' + str(abs(FWTM[0][1])*pixel_size) + ' mm')
    print( '')
    print( 'Point Source 2: ')
    print( ' FWHM x: ' + str(abs(FWHM[1][0])*pixel_size) + ' mm')
    print( ' FWHM y: ' + str(abs(FWHM[1][1])*pixel_size) + ' mm')
    print( ' FWTM x: ' + str(abs(FWTM[1][0])*pixel_size) + ' mm')
    print( ' FWTM y: ' + str(abs(FWTM[1][1])*pixel_size) + ' mm')
    print( '')
    print( 'Point Source 3: ')
    print( ' FWHM x: ' + str(abs(FWHM[2][0])*pixel_size) + ' mm')
    print( ' FWHM y: ' + str(abs(FWHM[2][1])*pixel_size) + ' mm')
    print( ' FWTM x: ' + str(abs(FWTM[2][0])*pixel_size) + ' mm')
    print( ' FWTM y: ' + str(abs(FWTM[2][1])*pixel_size) + ' mm')
    print( '')

import argparse

defVerbose = True
defSrcFile = None

if __name__ == '__main__' :
  usage = 'cdp 20171214 Extrinsic Spatial Resolution cameron.pain@austin.org.au'

  parser = argparse.ArgumentParser(description = usage)
  parser.add_argument('dicom_file', type = str, help = 'NM dicom file reconstructed axial')

  args = parser.parse_args()

  main(args.dicom_file )
#end if
