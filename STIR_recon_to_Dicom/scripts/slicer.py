#!/usr/bin/env python

# Remove excess black slices before and after the brain
# to remove Neurostat Alignment errors
# cdp 20170321


import numpy as n, matplotlib.pyplot as pyplot, pydicom
import os


def main(srcfile, ofile, verbose, threshold):
    ds = pydicom.read_file(srcfile)

    def slicer(A,a):
        p = []
        for x in A:
            if n.sum(x) >a:
                p.append(x)
        return n.array(p)

    H = slicer(ds.pixel_array,threshold)
    #Cuts the first 13 slices off the front and last off the back.
    H = H[0:len(H)-1,:,:]
    subSlice0 = n.zeros([n.shape(H[len(H)-1])[0]/2, n.shape(H[len(H)-1])[1]])
    subSlice1 = n.ones([n.shape(H[len(H)-1])[0]/2, n.shape(H[len(H)-1])[1]])
    subMask   = n.vstack([subSlice0,subSlice1])
    H[len(H)-1] = n.multiply(subMask,H[len(H)-1])
    ds.PixelData = H.tostring()
    ds[0x0009,0x101e].value = ds[0x0009,0x101e].value + '4'
    ds[0x0009,0x1045].value = ds[0x0009,0x101e].value
    ds[0x0008,0x0018].value = ds[0x0008,0x0018].value + '4'
    ds.SeriesDescription    = ds.SeriesDescription + '_neurostat'
    ds.NumberOfFrames = n.shape(H)[0]
    ds.NumberOfSlices = n.shape(H)[0]
    ds.SliceVector    = [i+1 for i in range(len(H))]
    ds.save_as( ofile)

#os.system('mv ' + str(ds.SeriesDescription)+'.dcm' + ' '+ str(ds.PatientName)+'/'+str(ds.SeriesDescription)+'.dcm')
    
    print 'dicom saved to ' + ofile





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
  parser.add_argument('threshold', type = int, help = 'Remove slice with integrated value below this value.')
                            
  args = parser.parse_args()
                                
  main(args.srcfile, args.ofile, args.verbose, args.threshold)
#end if



