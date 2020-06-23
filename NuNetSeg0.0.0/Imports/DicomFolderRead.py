#!/usr/bin/env python3
#
# Saves pixel data into a dicom file.
#
# cdp 20200212
#
import numpy as n
import matplotlib.pyplot as pyplot 
import pydicom
import os
import random
import sys
from matplotlib.widgets import Slider


def DicomFolderRead(relativePath):
    print('     DicomFolderRead():')
    start = time.time()
    dataFiles1 = os.popen('cd ' + relativePath +'; ls').read().split('\n')
    dataFiles = []
    print('         searching for .dcm files')
    for i in dataFiles1:
        if i!='':
            if (i[len(i)-3:] == 'dcm' or i[len(i)-3:] == 'DCM' or i[len(i)-3:] == 'ima' or i[len(i)-3:] == 'dcm' or i[len(i)-3:] == 'IMA') :
                dataFiles.append(  str(relativePath) + '/' + i)
    print('         .dcm files loaded.')
    end = time.time()
    print('         time elapsed ' + str(end - start) + ' seconds')
    return dataFiles
