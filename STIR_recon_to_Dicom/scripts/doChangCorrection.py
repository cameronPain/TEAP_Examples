#!/usr/bin/env python
#
# simple Chang correction
#
# C(x,y) = 1 / (1/M Sum{i=1,M} [exp(-mu.L(x,y,theta_i))])
#  M = number of projections used
#  L(x,y,theta_i) = distance from x,y to the edge of the attenuating body
#  mu = linear attenuation coefficient
#
# mu = 0.135 cm^-1 typically used, effective attenuation coefficient for both gammas from 111In
#
# gjok - 20120812
#

#
# define an attenuation mask
# define a line mask
# find the intersection of the line-mask with the attenuation mask
# generate the distance for each of the intersections
# assume the object is convex-hull
# use the maximum distance
#

import os, sys
from decimal import Decimal
import pydicom
import numpy
import scipy.ndimage
import matplotlib.pyplot as pyplot

defhuAir = -1000.0
defmuAir = 0.0
defSpectThresh = 0.05
defmuThresh = 0.05
defmuWater = 0.11   # effective attenuation coefficient for 111In
defStrtAngle = 0
defStopAngle = 360
defStepAngle = 5
defDebugPlots = False
defResultPlots = True


def dcmread(dcm_file, force = False) :
  """read dicom formatted file"""
  try:
    ds = pydicom.read_file(dcm_file)
  except pydicom.filereader.InvalidDicomError as e:
    if self.options.force:
      ds = pydicom.read_file(dcm_file, force = force)
    else:
      raise pydicom.filereader.InvalidDicomError("%s use force = 'true' if you are sure this is a dicom file" % e)
    #endif
  #end  try
  return ds
#end def dcmread

def dcmwrite(dcm_file, ds, fdata) :
  """write a dicom formatted file"""

  #
  # Convert the array to int16
  #
  fdmin = fdata.min()
  fdmax = fdata.max()
  if fdmin < 0 : raise AttributeError("negative values not supported")
 
  # Calculate new rescale slope and intercept
  dynamic_range = float(((2 ** 16) / 2.) - 1)
  dynamic_range = 1000.0
  islope = fdmax / dynamic_range
  iint = 0.0000
  
  idata = (fdata / fdmax * dynamic_range).astype(numpy.int16)    
  
  ds.RescaleIntercept = Decimal('%.4f' % (iint))
  ds.RescaleSlope = Decimal('%.6f' % (islope))   # doesn't like numbers that have too many decimal places
  ds.PixelData = idata.tostring()

  ds.StudyInstanceUID = ds.StudyInstanceUID + '1'
  print('Writing Dicom...')
  ds.save_as(dcm_file)
  print('Dicom written')
#end def dcmwrite

def projectVolume(slice) :
#
# project integrate along 90 and 270 deg from a given y voxel
#
  rv = numpy.zeros(slice.shape, dtype = slice.dtype)
  for y in range(0, slice.shape[1]) :
    rv[:, :, y] = slice[:, :, 0:y].sum(axis=2)
  #endfor
  return rv
#end def projectSlice

def calcChang(ds, spectThresh = defSpectThresh, muThresh = defmuThresh, resultPlots = defResultPlots, debugPlots = defDebugPlots) :
  deltaX = float(ds.PixelSpacing[0]) / 10.0
  frames = int(ds.NumberOfFrames)
  rows   = int(ds.Columns)
  cols   = int(ds.Rows)
  pixeldata = numpy.fromstring(ds.PixelData, dtype = numpy.int16).reshape(frames, rows, cols)
  
  RescaleSlope = 1.0
  RescaleIntercept = 0.0
  if hasattr(ds, 'RescaleSlope') : RescaleSlope = float(ds.RescaleSlope)
  if hasattr(ds, 'RescaleIntercept') : RescaleIntercept = float(ds.RescaleIntercept)
  
  spectVol = pixeldata.astype(numpy.float) * RescaleSlope + RescaleIntercept

  #
  # normalise
  #
  spectVol /= spectVol.max()
  nLTtissue = (spectVol < spectThresh)

  muVol    = numpy.zeros(spectVol.shape, dtype = numpy.float)
  muVol[:] = defmuWater
  muVol[nLTtissue] = 0.0
  
  bTissue = numpy.ones(spectVol.shape, dtype = numpy.int16)
  bTissue[nLTtissue] = 0

  bTissue = scipy.ndimage.morphology.binary_closing(bTissue, structure=numpy.ones((3,3,3)), iterations = 1)
  bTissue = scipy.ndimage.morphology.binary_fill_holes(bTissue)
  
  if True :
    print ds.PixelSpacing
    print 'deltaX = ', deltaX
    print 'frames = ', frames
    print 'rows = ', rows
    print 'cols = ', cols
    print 'muVol.shape = ', muVol.shape

    pyplot.figure()
    pyplot.imshow(spectVol[frames/2, :, :], cmap = pyplot.cm.hot)
    pyplot.colorbar()
    pyplot.title('specVol')
  #endif
  

  muChang = numpy.zeros(muVol.shape, dtype = muVol.dtype)
  angles  = range(defStrtAngle, defStopAngle, defStepAngle)
  for angle in angles :
    sys.stdout.write('angle = %d / %d\r' % (angle, defStopAngle))
    sys.stdout.flush()
    breshape = False
  
    muRot = scipy.ndimage.interpolation.rotate(muVol, angle, axes = (1, 2), reshape = breshape, cval = defmuAir)
    nRotLTtissue = (muRot < muThresh)
    muRot[nRotLTtissue] = defmuAir
  
#
# now project from each pixel within the threshold region
# then rotate that pixel back to the unrotated frame to add to the accumulator pixel
#
    muRotProject = projectVolume(muRot)
    muProject    = scipy.ndimage.interpolation.rotate(muRotProject, -angle, axes = (1, 2), reshape = breshape, cval = defmuAir)
    muChang += numpy.exp(-muProject * deltaX)
  #endfor

  muChang /= len(angles)
  for nslice in range(muVol.shape[0]) :
    muChangSlice = muChang[nslice, :, :]
    nLTchang = (muChangSlice <= 0)
    nGTchang = (muChangSlice > 0)
    muChangSlice[nGTchang] = 1.0 / muChangSlice[nGTchang]
    muChangSlice[nLTchang] = 0.0

    muChang[nslice, :, :] = muChangSlice
  #endfor
  muChang[nLTtissue] = 0.0
  
  if resultPlots :
    pyplot.figure()
    pyplot.imshow(muVol[frames/2, :, :] * deltaX, cmap = pyplot.cm.hot)
    pyplot.colorbar()
    pyplot.title('muVol*deltaX')

    pyplot.figure()
    pyplot.imshow(muChang[frames/2, :, :], cmap = pyplot.cm.hot)
    pyplot.colorbar()
    pyplot.title('Chang calculated attenuation')

    pyplot.show()
  
  #endif
#
# now scale by 1000 as we are dealing with NM storage
#
  return muChang * 1000
#enddef calcChang


def main(ifile, ofile) :

  ds = dcmread(ifile)

  sys.stdout.write('%s\n' % (ifile))
  chang = calcChang(ds)
  sys.stdout.write('\n')

  dcmwrite(ofile, ds, chang)
  print('file written?')
#enddef main

import argparse
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('ifile', type = str, help = 'The path to the input .dcm file.')
    parser.add_argument('ofile', type = str, help = 'The path to the output .dcm file.')
    args = parser.parse_args()
    main(args.ifile, args.ofile)
    #end if
