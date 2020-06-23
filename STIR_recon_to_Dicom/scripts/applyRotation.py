#!/usr/bin/env python

# Remove excess black slices before and after the brain
# to remove Neurostat Alignment errors
# cdp 20170321


import numpy as n, matplotlib.pyplot as pyplot, pydicom
import os
import scipy.ndimage as ndi

def main(srcfile, ofile, save):

    ds = pydicom.read_file(srcfile)

    pixelArray = ds.pixel_array
    pyplot.imshow(pixelArray[int(len(pixelArray)/2.0) - 10], cmap = pyplot.cm.gray, vmin = 0.0, vmax = n.amax(pixelArray))
    pyplot.show()


    pixelArray = ndi.rotate(pixelArray, -90, axes=(1,2), reshape=False)

    pyplot.imshow(pixelArray[int(len(pixelArray)/2.0)-10], cmap = pyplot.cm.gray, vmin = 0.0, vmax = n.amax(pixelArray))
    pyplot.show()



    if save:
        ds.PixelData = pixelArray.tostring()
        ds.save_as(ofile)






import argparse

defVerbose = True
defSrcFile = None

if __name__ == '__main__' :
  usage = '\nusage: %prog [options] srcdir\n\tproblems to graeme.okeefe@austin.org.au\n\teg. %prog NM-dicom-file'
        
  parser = argparse.ArgumentParser(description = usage)
            
  group = parser.add_mutually_exclusive_group()
  group.add_argument('--verbose',   dest = 'verbose', default = defVerbose, action = 'store_true', help = 'default = %s' % ('True' if defVerbose else 'False'))
  group.add_argument('--noverbose', dest = 'verbose', default = defVerbose, action = 'store_false')
                        
  parser.add_argument('srcfile', type = str, help = 'NM dicom file')
  parser.add_argument('ofile',   type = str, help = 'The name of the output file (include the file extension).')
  parser.add_argument('--save',  dest = 'save', default = False, action = 'store_true', help = 'Set this flag to save the ofile.')
  args = parser.parse_args()
                                
  main(args.srcfile, args.ofile, args.save)
#end if



