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


def saveDicomFile(data, sliceTemplate, output_dir, modality, factor = 1, organ_name = ''):
    templateFiles = sliceTemplate[:len(data)]
    if type(factor)== int:
        factorX = factor
        factorY = factor
        factorZ = factor
    elif type(factor) == list:
        factorX = factor[2]
        factorY = factor[1]
        factorZ = factor[0]
    else:
        print('\n\n\n')
        print('---------  ERROR   ---------')
        print('Can not define the single or multidimensional rebinning factor in the saveDicomFile() function. Check that you have formatted the rebinning_factor input correctly.')
        sys.exit()
    if modality == 'CT':
        #data      = data[::-1,:]
        ds        = pydicom.read_file(sliceTemplate[0],force = True)
        px        = float(ds.PixelSpacing[0])
        dim       = n.shape(data)
        nofSlices = dim[0]
        nofCol    = dim[2]
        nofRow    = dim[1]
        print(dim)
        sliceThickness = float(ds.SliceThickness)*factorZ
        x_px           = px*factorX
        y_px           = px*factorY
        # generate new UIDs
        UR1 = ''
        UR2 = ''
        UR3 = ''
        UR4 = ''
        for i in range(5):
            UR1 = UR1 + str(int(n.random.random()*100))
            UR2 = UR2 + str(int(n.random.random()*100))
            UR3 = UR3 + str(int(n.random.random()*100))
            UR4 = UR4 + str(int(n.random.random()*100))
        frameUID        = ds.FrameOfReferenceUID[:-5] + UR2
        studyUID        = ds.StudyInstanceUID[:-5]    + UR3
        seriesUID       = ds.SeriesInstanceUID[:-5]   + UR4
        #Adjust the slice position accordingly.
        startPatientPosition = float(pydicom.read_file(sliceTemplate[-1], force = True).ImagePositionPatient[2])
        data = n.multiply(data, 1024*2)
        print('shape of saved data: ', n.shape(data))
        #Save the data.
        for i in range(len(data)):
            dsSlice                         = pydicom.read_file(templateFiles[i], force = True)
            dsSlice.ImagePositionPatient[2] = str(startPatientPosition + i * sliceThickness)
            dsSlice.SeriesInstanceUID       = seriesUID
            #dsSlice.StudyInstanceUID       = studyUID
            dsSlice.FrameOfReferenceUID     = frameUID
            dsSlice.PixelData               = data[i].astype(n.int16).tostring()
            dsSlice.PixelSpacing[0]         = x_px
            dsSlice.PixelSpacing[1]         = y_px
            dsSlice.SpacingBetweenSlices    = -1*sliceThickness
            dsSlice.SliceThickness          = sliceThickness
            dsSlice.Rows                    = nofRow
            dsSlice.Columns                 = nofCol
            dsSlice.SliceLocation           = 0+sliceThickness*i
            dsSlice.SeriesDescription       = dsSlice.SeriesDescription + '_rebinned_'+ organ_name + '_' + str((factorZ,factorY,factorX))
            dsSlice.save_as(output_dir + str(i) + '.dcm')
        print('Data saved to '  + output_dir)


    elif modality == 'PT':
        #data      = data[::-1,:]
        ds        = pydicom.read_file(sliceTemplate[0], force = True)
        px        = float(ds.PixelSpacing[0])
        dim       = n.shape(data)
        nofSlices = dim[0]
        nofCol    = dim[2]
        nofRow    = dim[1]
        sliceThickness = float(ds.SliceThickness)*factorZ
        x_px           = px*factorX
        y_px           = px*factorY
        # generate new UIDs
        UR2 = ''
        UR3 = ''
        UR4 = ''
        for i in range(5):
            UR2 = UR2 + str(int(n.random.random()*9))
            UR3 = UR3 + str(int(n.random.random()*9))
            UR4 = UR4 + str(int(n.random.random()*9))
        frameUID        = ds.FrameOfReferenceUID[:-5] + UR2
        studyUID        = ds.StudyInstanceUID[:-5]    + UR3
        seriesUID       = ds.SeriesInstanceUID[:-5]   + UR4

        #Adjust the slice position accordingly.
        startPatientPosition = float(pydicom.read_file(sliceTemplate[0], force = True).ImagePositionPatient[2])
        #Save the data.
        for i in range(len(data)):
            dsSlice                         = pydicom.read_file(templateFiles[i], force = True)
            dsSlice.ImagePositionPatient[2] = str(startPatientPosition + i * sliceThickness)
            print(str(startPatientPosition + i * sliceThickness))
            dsSlice.SeriesInstanceUID       = seriesUID
            #dsSlice.StudyInstanceUID       = studyUID
            dsSlice.FrameOfReferenceUID     = frameUID
            dsSlice.PixelData               = data[i].astype(n.int16).tostring()
            dsSlice.PixelSpacing[0]         = x_px
            dsSlice.PixelSpacing[1]         = y_px
            dsSlice.SpacingBetweenSlices    = -1*sliceThickness
            dsSlice.SliceThickness          = sliceThickness
            dsSlice.Rows                    = nofRow
            dsSlice.Columns                 = nofCol
            dsSlice.SliceLocation           = 0+sliceThickness*i
            dsSlice.SeriesDescription       = dsSlice.SeriesDescription + '_rebinned_' + str((factorZ,factorY,factorX))
            dsSlice.save_as(output_dir + str(i) + '.dcm')
        print('Data saved to '  + output_dir)

    elif modality == 'NM':
        ds        = pydicom.read_file(sliceTemplate[0],force = True)
        px        = float(ds.PixelSpacing[0])
        dim       = n.shape(data)
        nofSlices = dim[0]
        nofCol    = dim[2]
        nofRow    = dim[1]
        sliceThickness = float(ds.SliceThickness)*factorZ
        x_px           = px*factorX
        y_px           = px*factorY
        ds.PixelData       = data.astype(n.int16).tostring()
        ds.PixelSpacing[0] = x_px
        ds.PixelSpacing[1] = y_px
        ds.SpacingBetweenSlices = -1*sliceThickness
        ds.SliceThickness = sliceThickness
        ds.Rows            = nofRow
        ds.Columns         = nofCol
        ds.NumberOfFrames  = len(data)
        ds.NumberOfSlices  = len(data)
        ds.InstanceNumber  = str(int(ds.InstanceNumber) + 1)
    
        # generate new UIDs
        UR2 = ''
        UR3 = ''
        UR4 = ''
        for i in range(5):
            UR2 = UR2 + str(int(n.random.random()*9))
            UR3 = UR3 + str(int(n.random.random()*9))
            UR4 = UR4 + str(int(n.random.random()*9))
        ds.FrameOfReferenceUID    = ds.FrameOfReferenceUID[:-5] + UR2
# ds.StudyInstanceUID       = ds.StudyInstanceUID[:-5]    + UR3
        ds.SeriesInstanceUID      = ds.SeriesInstanceUID[:-5]   + UR4
        ds.SeriesDescription = ds.SeriesDescription + '_rebinned_' + str((factorZ,factorY,factorX))
        ds.save_as(output_dir + 'output_file.dcm')
        print('Data saved to '  + output_dir)
    else:
        print('\n\n\n')
        print('---------  ERROR   ---------')
        print('Undefined modality handed to the saveDicomFile() function. It takes either CT, NM or PT. It was given the modality ' + modality)
        sys.exit()
