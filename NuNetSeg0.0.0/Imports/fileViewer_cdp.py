#!/usr/bin/env python3
#
# View a DICOM file dataset as an animation or a click through.
#
# cdp 20190806
#
import pydicom
import numpy as n
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
import sys
from   matplotlib.widgets import Slider
from   DicomFolderRead import DicomFolderRead as DFR

#Import viewer function. The difference is this takes an image matrix and the other takes a .dcm file.


def orderDCMFiles(file_list):
    numerical_list = [int(file_list[i].split('/')[-1][:-4]) for i in range(len(file_list))]
    ordered_strNum_list = n.sort(numerical_list).astype(str)
    ordered_file_list   = [ordered_strNum_list[i] + '.dcm' for i in range(len(ordered_strNum_list))]
    
    ordered_paths  = ['/'.join(file_list[i].split('/')[:-1]) + ordered_file_list[i] for i in range(len(file_list))]
    return ordered_paths

def fileViewer(imageMatrix):
    fig,(im_ax) = pyplot.subplots(1, 1, figsize=(21,11))
    max_val     = n.amax(imageMatrix)
    image       = im_ax.imshow(imageMatrix[0], cmap = pyplot.cm.binary, vmin = 0.0, vmax = max_val)
    sliderAx    = pyplot.axes([0.1,0.1,0.18,0.02])
    sliceSlider = Slider(sliderAx, 'Slice', 0, len(imageMatrix)-1, valstep = 1)
    def change_slice(val):
        new_val = int(val)
        image.set_data(imageMatrix[new_val])
        fig.canvas.draw()
    sliceSlider.on_changed(change_slice)
    pyplot.show()






#Main viewer function
def File_Viewer(srcfile) :
    ds          = DFR(srcfile)
    ds_ordered  = orderDCMFiles(ds)
    imageMatrix = []
    for file in ds_ordered:
        imageMatrix.append(pydicom.read_file(file).pixel_array)
    imageMatrix = n.array(imageMatrix)
    print(n.shape(imageMatrix))
    fig,(im_ax) = pyplot.subplots(1, 1, figsize=(21,11))
    max_val     = n.amax(imageMatrix)
    image       = im_ax.imshow(imageMatrix[0], cmap = pyplot.cm.binary, vmin = 0.0, vmax = max_val)
    sliderAx    = pyplot.axes([0.1,0.1,0.18,0.02])
    sliceSlider = Slider(sliderAx, 'Slice', 0, len(imageMatrix)-1, valstep = 1)
    def change_slice(val):
        new_val = int(val)
        image.set_data(imageMatrix[new_val])
        fig.canvas.draw()
    sliceSlider.on_changed(change_slice)
    pyplot.show()





import argparse
if __name__ == '__main__' :
  usage = 'Cameron Pain (cameron.pain@austin.org.au): Simple image viewer for TEAP.'
  parser = argparse.ArgumentParser(description = usage)
  parser.add_argument('srcfile', type = str, help = 'NM dicom file')
  args = parser.parse_args()
  File_Viewer(args.srcfile)