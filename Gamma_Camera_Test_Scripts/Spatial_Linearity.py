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
import scipy.ndimage as ndi

def sumROI(Image, axis):
    data = []
    for i in range(int(len(Image[0])/4)-1):
        data.append(n.sum(Image[:,4*i:(i+1)*4], axis = axis))
    return n.array(data).transpose()


def localiseLines(Image, orientation):
    if orientation == 'x':
        proj = n.sum(Image,axis=1)
    elif orientation == 'y':
        proj = n.sum(Image,axis=0)
    else:
        print('Define an orientation. Default to x.')
    px         = n.arange(len(proj))
    pxRange    = n.arange(0,len(proj), 0.01)
    params,cov = curve_fit(sinusoid, px, proj, p0=[(2*n.pi/14.0), 3, 2000, 3000] )
    fit        = sinusoid(pxRange, *params)
    pyplot.scatter(px, proj, color='b')
    pyplot.plot(pxRange, fit)
    pyplot.show()


    minima = []
    for i in range(1,9):
        minima.append(  int((2*n.pi*i- params[1]) / (params[0]))  )
    peaks = []
    for i in range(len(minima)-1):
        peaks.append(     Image[minima[i]:minima[1+i],:]     )
    return peaks


def sinusoid(x, w, phi, A, C):
    if type(x)!=n.ndarray and type(x)!=list:
        return A*n.sin(w*x - phi) + C
    else:
        data = []
        for i in x:
            data.append( sinusoid(i, w, phi, A, C))
        return n.array(data)


def gaussian(x, x0, A, sigma, C):
    if type(x)!=n.ndarray and type(x)!=list:
        return A*n.exp(-((x-x0)**2)/(2*(sigma**2))) + C
    else:
        data = []
        for i in x:
            data.append(gaussian(i, x0, A, sigma, C))
        return n.array(data)


def main(srcfile):
    ds        = pydicom.read_file(srcfile)
    pixelData = ds.pixel_array

    dim       = n.shape(pixelData)
    ROI1      = pixelData[ int(5*dim[1]/8):int(6*dim[1]/8.0), int(2*dim[0]/8):int(3*dim[0]/8) ]
    ROI2      = pixelData[ int(2*dim[1]/8):int(3*dim[1]/8), int(2*dim[0]/8):int(3*dim[0]/8) ]
    ROI3      = pixelData[ int(2*dim[1]/8):int(3*dim[1]/8), int(5*dim[1]/8):int(6*dim[1]/8.0) ]
    ROI4      = pixelData[ int(5*dim[1]/8):int(6*dim[1]/8.0), int(5*dim[1]/8):int(6*dim[1]/8.0) ]
    
    
    summedROI1 = sumROI(ROI1, 1)
    pyplot.imshow(summedROI1)
    pyplot.show()
    
    pyplot.imshow(pixelData)
    pyplot.show()
    pyplot.imshow(ROI1)
    pyplot.show()
    pyplot.imshow(ROI2)
    pyplot.show()
    pyplot.imshow(ROI3)
    pyplot.show()
    pyplot.imshow(ROI4)
    pyplot.show()

    lines1 = localiseLines(summedROI1,'x')
    lines2 = localiseLines(ROI2,'y')
    
    centroidLocStdev = []
    for lineProfile in lines1:
        pyplot.imshow(lineProfile, cmap = pyplot.cm.binary, vmin = 0.0, vmax = n.amax(lineProfile))
        pyplot.show()
        maxima = []
        for i in range(len(lineProfile)):
            linePx, lineRange = n.arange(len(lineProfile)), n.arange(0,len(lineProfile), 0.01)
            lineData          = lineProfile[:,i]
            params, cov = curve_fit(gaussian, linePx, lineData, p0=[len(lineData)/2, n.amax(lineData), len(lineData)/4, 0.05*n.amax(lineData)])
            fitData     = gaussian(lineRange, *params)
                #if (n.sum(abs(cov)) >= 2*(params[1]+params[3]) or (params[1]+params[3]) <= 0.75*n.amax(lineProfile)):
                #print(n.sum(abs(cov)))
                #print(0.75*n.amax(lineProfile), (params[1]+params[3]))
                #pyplot.plot(lineRange, fitData, '--', color='b')
                #pyplot.scatter( n.arange(len(lineProfile)), lineData)
                #pyplot.show()
            maxima.append( params[0] )
        centroidLocStdev.append([n.std(maxima)])
    print('')
    print('')
    print('Differential Linearity (Using the thickest line pairs): ')
    print('      mean: ' + str(n.average(centroidLocStdev)) + ' mm')
    print('     stdev: ' + str(n.std(centroidLocStdev))     + ' mm')
    print('       max: ' + str(n.amax(centroidLocStdev))    + ' mm')
    print('')
    print('')




    print('')
    print('')
    print('ROI 1 spatial linearity.')



import argparse
if __name__ == '__main__' :
  usage = 'Written by Cameron Pain. Calculates the integral and differential uniformities on the UFOV and CFOV of a flood image. The image post processing and analysis is done according to the NEMA NU1 protocol.'
  parser = argparse.ArgumentParser(description = usage)
  parser.add_argument('srcfile', type = str, help = 'Specify the path to the .dcm file containing the uniformity images.')
  args = parser.parse_args()
  main(args.srcfile)
#end ifls
