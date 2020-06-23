#!/usr/bin/env python3
#
# Measure the planar sensitivity from dicom file of raw data
#
# cdp 20171031
#


import pydicom, numpy as n, matplotlib.pyplot as pyplot, scipy.ndimage as ndi, scipy.fftpack as fftp
#from scipy.misc import imresize
import os
from scipy.optimize import curve_fit as curve_fit


def gaussianFit(t,t0,A,sigma):
    return A*n.exp(  -((t-t0)**2)/(2*(sigma**2))   )

def gaussianPlot(T,params):
    data = []
    for i in T:
        data.append(gaussianFit(i,*params))
    return n.array(data)


def zeroRowRemoval(Image, threshold):
    data = []
    for i in Image:
        if n.sum(i) >= threshold:
            data.append(i)
    data = n.array(data)
    data1 = []
    transData = n.transpose(data)
    for i in transData:
        if n.sum(i) >= threshold:
            data1.append(i)
    return n.array(data1)


def NEMACentroidLocator(profileX,profileY):
    pixelNumbersX = n.add(n.arange(len(profileX)),1)
    pixelNumbersY = n.add(n.arange(len(profileY)),1)
    CentroidLocationX = n.sum(n.multiply(pixelNumbersX, profileX))/(n.sum(profileX))
    CentroidLocationY = n.sum(n.multiply(pixelNumbersY, profileY))/(n.sum(profileY))
    return [CentroidLocationX, CentroidLocationY]



def maxPixelIndex(Image):
    maxPixel = n.amax(Image)
    for i in range(len(Image)):
        for j in range(len(Image[0])):
            if Image[i,j] == maxPixel:
                maxCoordinates = [i,j]
    return maxCoordinates


def locateCentroids(Array, plot, pixelSize):
    Centroids = []
    NEMACentroids = []
    
    for sliceIndex in range(len(Array)):
        
        if sliceIndex == 0:
            nofPixels = int((40/pixelSize)/2)
        elif sliceIndex == 1:
            nofPixels = int((60/pixelSize)/2)
        else:
            nofPixels = int((80/pixelSize)/2)
  
        slice = Array[sliceIndex]
        
        index = maxPixelIndex(slice)
        #imageROI = slice[ index[0]-nofPixels:index[0]+nofPixels, index[1]-nofPixels :index[1]+nofPixels ]
        #pyplot.imshow,cmap=pyplot.cm.jet, vmin = 0.0, vmax = n.amax(imageROI))
        #pyplot.show()
        xPixels = n.arange(len(slice[0]))
        yPixels = n.arange(len(slice))
        projectionY = n.sum(slice,axis=1)
        projectionX = n.sum(slice,axis=0)



        # remove data less than half the maximum
        xDataTruncated   = []
        xPixelsTruncated = []
        yDataTruncated   = []
        yPixelsTruncated = []
        xKeepThreshold = 0.15*n.amax(projectionX)
        yKeepThreshold = 0.15*n.amax(projectionY)

        for i in range(len(projectionX)):
            if projectionX[i] >= xKeepThreshold:
                xDataTruncated.append(  projectionX[i]    )
                xPixelsTruncated.append( xPixels[i])
            if projectionY[i] >= yKeepThreshold:
                yPixelsTruncated.append(yPixels[i])
                yDataTruncated.append( projectionY[i])
    
        xDataTruncated   = n.array(xDataTruncated)
        yDataTruncated   = n.array(yDataTruncated)
        yPixelsTruncated = n.array(yPixelsTruncated)
        xPixelsTruncated = n.array(xPixelsTruncated)
        #Fit Gaussian models to truncated data
        

        NemaCentroids = [n.sum(n.multiply(xDataTruncated,xPixelsTruncated))/n.sum(xDataTruncated),n.sum(n.multiply(yDataTruncated,yPixelsTruncated))/n.sum(yDataTruncated)]
        
        averageCentroidX = n.sum(xPixelsTruncated)/float(len(xPixelsTruncated))
        averageCentroidY = n.sum(yPixelsTruncated)/float(len(yPixelsTruncated))
        
        paramsX, cov = curve_fit(gaussianFit, xPixelsTruncated, xDataTruncated, [ averageCentroidX, 50000,50])
        paramsY, cov = curve_fit(gaussianFit, yPixels, projectionY, [averageCentroidY, 50000 ,50])
        
        pRange  = n.arange(0,len(xPixels),0.05)
        dataGaussianX = gaussianPlot(pRange,paramsX)
        dataGaussianY = gaussianPlot(pRange,paramsY)


        if plot:
            fig, (ax0) = pyplot.subplots(1,1,figsize=(21,10))
            ax0.plot(pRange, dataGaussianX, color='b')
            ax0.plot(pRange, dataGaussianY, color='r')
            ax0.scatter(xPixelsTruncated, xDataTruncated, color='b', label = 'x projection')
            ax0.scatter(yPixelsTruncated, yDataTruncated, color='r', label = 'y projection')
            ax0.set_title('Photopeak ' + str(sliceIndex))
            ax0.set_xlabel('pixel coordinate')
            ax0.set_ylabel('Counts')
            pyplot.legend(loc=0)
            pyplot.show()
        
        
        truncatedImageCentroids = NEMACentroidLocator(projectionX, projectionY)
        sliceNEMACentroids = n.add(index, n.subtract(truncatedImageCentroids, [nofPixels+1, nofPixels+1]))
        NEMACentroids.append(  NemaCentroids  )
        Centroids.append([paramsX[0], paramsY[0]])
    return Centroids, NEMACentroids


def main(srcfolder, plot):

# Make a list of all the file names in the folder of interest. (Each slice saved as an individual
# dicom for some reason).
    if srcfolder[-1] == '/':
        pass
    else:
        srcfolder += '/'
    dataFiles1 = os.popen('cd ' + srcfolder +'; ls').read().split('\n')[:-1]
    dataFiles  = []
    for i in dataFiles1:
        if (i[-4:] == '.IMA' or i[-4:] == '.dcm' or i[-4:] == '.DCM' or i[-4] == '.ima'):
            dataFiles.append(srcfolder + i)
    print('Using datafiles: ')
    for i in dataFiles:
        print(i)
# Using all the file names, make a empty set 'p' and collect the pixel data from each dicom folder into
# a single array. So I've taken each slice and put them together to make a volume.

    p = []
    for i in dataFiles:
        ds = pydicom.read_file(i)
        data = ds.pixel_array
        p.append(data)

# p now contains all 9, 3 X N X N arrays for each energy of each acquisition.


    pixelSpacing = float(ds.PixelSpacing[0])
    distances = []
    NEMAdistances = []
    load_string = '. processing'
    for dataArray in p:
        centroids = locateCentroids(dataArray, plot, pixelSpacing)
        if len(centroids[0]) == 3:
            distances.append(  n.sqrt(     (centroids[0][0][0]-centroids[0][1][0])**2 + (centroids[0][0][1] - centroids[0][1][1])**2   )   )
            distances.append(  n.sqrt(     (centroids[0][0][0]-centroids[0][2][0])**2 + (centroids[0][0][1] - centroids[0][2][1])**2   )   )
            distances.append(  n.sqrt(     (centroids[0][2][0]-centroids[0][1][0])**2 + (centroids[0][2][1] - centroids[0][1][1])**2   )   )
            NEMAdistances.append(  n.sqrt(     (centroids[1][0][0]-centroids[1][1][0])**2 + (centroids[1][0][1] - centroids[1][1][1])**2   )   )
            NEMAdistances.append(  n.sqrt(     (centroids[1][0][0]-centroids[1][2][0])**2 + (centroids[1][0][1] - centroids[1][2][1])**2   )   )
            NEMAdistances.append(  n.sqrt(     (centroids[1][2][0]-centroids[1][1][0])**2 + (centroids[1][2][1] - centroids[1][1][1])**2   )   )
            print(load_string, end='\r')
            load_string = '.' + load_string
        elif len(centroids[0]) == 2:
            distances.append(  n.sqrt(     (centroids[0][0][0]-centroids[0][1][0])**2 + (centroids[0][0][1] - centroids[0][1][1])**2   )   )
            NEMAdistances.append(  n.sqrt(     (centroids[1][0][0]-centroids[1][1][0])**2 + (centroids[1][0][1] - centroids[1][1][1])**2   )   )
            print(load_string, end='\r')
            load_string = '.' + load_string
        elif len(centroid[0]) == 1:
            print('ERROR: Only one photopeak found in the .dcm files.')
            sys.exit()
        else:
            print('Warning: More than 3 photopeaks found. Only analysing the first three.')
            distances.append(  n.sqrt(     (centroids[0][0][0]-centroids[0][1][0])**2 + (centroids[0][0][1] - centroids[0][1][1])**2   )   )
            distances.append(  n.sqrt(     (centroids[0][0][0]-centroids[0][2][0])**2 + (centroids[0][0][1] - centroids[0][2][1])**2   )   )
            distances.append(  n.sqrt(     (centroids[0][2][0]-centroids[0][1][0])**2 + (centroids[0][2][1] - centroids[0][1][1])**2   )   )
            NEMAdistances.append(  n.sqrt(     (centroids[1][0][0]-centroids[1][1][0])**2 + (centroids[1][0][1] - centroids[1][1][1])**2   )   )
            NEMAdistances.append(  n.sqrt(     (centroids[1][0][0]-centroids[1][2][0])**2 + (centroids[1][0][1] - centroids[1][2][1])**2   )   )
            NEMAdistances.append(  n.sqrt(     (centroids[1][2][0]-centroids[1][1][0])**2 + (centroids[1][2][1] - centroids[1][1][1])**2   )   )
            print(load_string, end='\r')
            load_string = '.' + load_string

    print('\n')
    MWSR = n.sum(distances)/float(len(NEMAdistances))
    MWSR_NEMA = n.sum(NEMAdistances)/float(len(NEMAdistances))


    print( '############')
    print( ' Multiple Window Spatial Registration Offset from Gaussian fit: ' + str(MWSR) + ' mm')
    print( ' Multiple Window Spatial Registration Offset from NEMA centroids: ' + str(MWSR_NEMA) + ' mm')
    print( '############')


import argparse

defVerbose = True
defSrcFile = None

if __name__ == '__main__' :
  usage = 'cdp 20171031 cameron.pain@austin.org.au'
  parser = argparse.ArgumentParser(description = usage)

  parser.add_argument('srcfolder', type = str, help = 'Folder containing the 9 NM dicom files of MWSR data.')
  parser.add_argument('--plot', dest = 'plot', default = False, action = 'store_true', help = 'Set this tag to plot the projections.')
  
  args = parser.parse_args()
                                
  main(args.srcfolder, args.plot)
#end if



