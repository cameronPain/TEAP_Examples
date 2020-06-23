#!/usr/bin/env python3
#
#
#
#
import matplotlib.pyplot as pyplot
import numpy as n
import os
import pydicom
import scipy.ndimage as ndi
import time
import datetime
import sys
from scipy.ndimage.morphology import binary_dilation
workingDir = os.popen('pwd').read().split('\n')[0]
sys.path.append(workingDir + '/Imports/')
print(workingDir + '/Imports/' + ' appended to sys.path')
start = time.time()
print('     removing FutureWarning prints...')
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow
import tensorflow.keras.models
from keras import backend as K
#local imports
from saveDicomFile_cdp    import saveDicomFile
from fileViewer_cdp       import fileViewer
from get_boundary_cdp     import get_contour_library, concatenate_contours, collect_single_contour, create_dicom_coordinates_contour, remove_zero_padding, smoothROI, smoothROI_KeepHoles, remove_zero_padding_image
from inputDataParser_cdp  import zoom_to_size, DFR, sum_rebin, mean_rebin, transform_to_N, pad_to_N, is_odd, get_CT_data, get_CTandNM_data, get_dicom_header_info_CT, get_projection_bounds, get_ROI_Z_CoM, tensorflow_format, Organ_Localiser_CT, Organ_Localiser_CTandNM, generate_NM_Crop_Coordinates, fit_to_template_3D, normalise, _thresholdPrediction, apply_window, get_gaussian_kernel

class Segmentation_Process:
    def __init__(self):
        #print('Processing data for input to NeuNets.')
        self.Base_Localiser_Dim    = [64,64,64] # Hard coded to create a [64,64,64] rebin to use for all segmentation processes.
        self.Dicom_Contour_Library = [] #Catch the contours made in here.
        self.Mask_Library          = []
        #NOF_SLICES      = 256 # Hard coded to reshape to a 256x512x512 image either by padding/trimming or stretching/squeezing.
        #self.InputDataParser      = InputDataParser()
    
    def ProcessDataForInput(self, Input_Data_Dir):
        print('Gathering CT_data.')
        self.CT_data, self.Flip   = get_CT_data(Input_Data_Dir)
        print('Acquired CT_data')
        print('Gathering localisation data.')
        self.Localisation_Rebin   = ndi.zoom(self.CT_data, n.divide(self.Base_Localiser_Dim, n.shape(self.CT_data)))
        print('Gathering CT_header.')
        self.Dicom_Header         = get_dicom_header_info_CT(Input_Data_Dir)
        print('Collected CT_header.')
        #print('Dataset mirrored axially to fit convention: ', Flip)
        #CT_trim, padded = transform_to_N(n.divide(CT_data,n.amax(CT_data)),NOF_SLICES)
        print('Shape of input data: ', n.shape(self.CT_data))

#, padded #The prepared dataset used for NeuNet Segmentation (n.ndarray), the dataset from the dicom files (n.ndarray), The header of the first dcm image in the stack (pydicom.dataset.Dataset), whether datawas flipped along z to match convention (bool), whether data was padded (True) or whether it was stretched (False) (bool).

#def PerformNeuNetSegmentation(self, Input_Data, Localisation_Rebin_Data, Localisation_Model, Segmentation_Model, Localisation_Reshape_Factor, Segmentation_Input_Dim, Header, Prediction_Threshold = 0.5, Localisation_Threshold = 0.3, Localisation_Buffer = [[5,5],[5,5],[5,5]], Flip = True, renormalisation_factor = 1.0):

    def PerformNeuNetSegmentation(self, NeuNetSeg):
        nof_segmentations = len(NeuNetSeg.settings.localisation_models)
        for i in range(nof_segmentations):
            print('     ' + NeuNetSeg.settings.localisation_models[i])
            print('     Organ localisation:')
            Crop_Coods  = Organ_Localiser_CT(self.CT_data, self.Localisation_Rebin,  NeuNetSeg.settings.localisation_shape[i], NeuNetSeg.localisation_models[i], Localisation_Threshold = NeuNetSeg.settings.localisation_threshold[i], buffer = NeuNetSeg.settings.localisation_buffer[i])
            print('     Trimming data with crop coordinates.')
            CT_Trim         = self.CT_data[ Crop_Coods[0]    : Crop_Coods[1]     , Crop_Coods[2]     : Crop_Coods[3]     , Crop_Coods[4]    : Crop_Coods[5]        ]
            print('     Zooming the data to fit NeuNet input.')
            CT_Trim      = normalise(CT_Trim)#Normalise each NeuNet input rather than the whole image and then cropping out bits. This is in line with how the NeuNets are trained.
            #fileViewer(CT_Trim)
            NeuNetInput  = ndi.zoom(CT_Trim , n.divide( n.array(NeuNetSeg.settings.segmentation_shape[i]), n.shape(CT_Trim) ))
            NeuNetInput  = normalise(NeuNetInput) #After zooming with ndi.zoom you lose peak normalisation, so renormalise after each zoom.
            #Reshape back into the crop coords. Best to do this before thresholding in case the interpolation causes that shadow effect from adjacent slices.
            print('     Segmenting.')
            prediction   = NeuNetSeg.segmentation_models[i].predict(tensorflow_format(NeuNetInput))
            #Reshape back into the crop coords. Best to do this before thresholding in case the interpolation causes that shadow effect from adjacent slices.
            prediction   = ndi.zoom(prediction[0,:,:,:,0], n.divide(n.shape(CT_Trim), n.array(NeuNetSeg.settings.segmentation_shape[i])    ))
            smoothing_kernel  = get_gaussian_kernel(5, 0.5 + 0.25 * NeuNetSeg.settings.smoothing_factor[i])
            prediction   = ndi.convolve(prediction, smoothing_kernel)
            predictionTh = _thresholdPrediction(prediction, NeuNetSeg.settings.segmentation_threshold[i])
            print('     Post processing prediction.')
            self.Mask_Library.append(predictionTh)
            print('     Collecting contours.')
            boundary        = []
            boundary_kernel = n.array([[0,1,0],[1,1,1],[0,1,0]])
            for slice in predictionTh:
                boundary.append(n.multiply(slice, binary_dilation(slice == 0, boundary_kernel).astype(int)))
            boundary        = n.array(boundary)
            template_original  = n.zeros(n.shape(self.CT_data))
            template_original[ Crop_Coods[0]: Crop_Coods[1] , Crop_Coods[2]: Crop_Coods[3] , Crop_Coods[4]: Crop_Coods[5]] = boundary
            if NeuNetSeg.segmentation_process.Flip:
                template_original  = template_original[::-1]
            contour_library        = get_contour_library(template_original)
            print('     Transforming contours to dicom patient coordinate system.')
            #contour_library        = remove_zero_padding(contour_library, Original_CT_Data)
            self.Dicom_Contour_Library.append( create_dicom_coordinates_contour(contour_library, NeuNetSeg.segmentation_process.Dicom_Header, Flipped = NeuNetSeg.segmentation_process.Flip) )
            print('--------------------------------------------------------------------------------')


import argparse
if __name__ == '__main__':
    usage  = 'Cameron Pain (cameron.pain@austin.org.au): Python3 dicom node for neural network segmentation of WBCT. Listens on a specified port at a specified IP address (default set to my 172.27.51.140 workstation).'
    parser = argparse.ArgumentParser(description = usage)
    parser.add_argument('--AETitle', dest = 'AETitle', default = 'WBCT_Segmentation', type = str , help = 'The AE title the server.')
    parser.add_argument('--ip',      dest = 'ip',      default = '172.27.51.140',  type = str, help = 'The IP address for your server.')
    parser.add_argument('--port',    dest = 'port',    default = 1201,             type = int, help = 'The port to run your server on.')
    args = parser.parse_args()
    main(args.AETitle, args.ip, args.port)
