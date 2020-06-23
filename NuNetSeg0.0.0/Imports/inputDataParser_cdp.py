#!/usr/bin/env python3
#
# An import file where I define the functions necessary to parse the input data appropriately. This will include rebinning, localising, croping, zooming etc.
#
# cdp 20200327
#
import numpy as n
import matplotlib.pyplot as pyplot
from scipy.optimize import curve_fit as curve_fit
import os
import pydicom
from matplotlib.widgets import Slider
import scipy.ndimage as ndi
import time
import datetime
import sys
from scipy.ndimage.morphology import binary_dilation
start = time.time()
from fileViewer_cdp import fileViewer


def gaussian(x,y,z,sigma,xyz0):
    return n.exp(-((x-xyz0)**2)/(2*(sigma**2)))*n.exp(-((y-xyz0)**2)/(2*(sigma**2)))*n.exp(-((z-xyz0)**2)/(2*(sigma**2)))

def get_gaussian_kernel(shape, sigma):
    kernel = []
    for i in range(shape):
        kernel_slice = []
        for j in range(shape):
            kernel_row = []
            for k in range(shape):
                kernel_row.append( gaussian(i,j,k,sigma,int(shape/2)))
            kernel_slice.append(kernel_row)
        kernel.append(kernel_slice)
    kernel = n.array(kernel)
    kernel = n.divide(kernel, n.sum(kernel))
    return kernel


#Pulling out raw pixel data. The values are defined as HU + 1024 such that vacuum is 0 and water = 1024
def apply_window(data, upper_threshold = 2000): #Setting upper threshold to >1500 will suppress metal artefacts (Dental and screws etc have messed things up before.)
    low_data = (data <= upper_threshold)*data
    threshold_data = (data > upper_threshold).astype(int)*n.amax(low_data)
    windowed_data = n.add(low_data, threshold_data)
    norm_data     = n.divide(windowed_data, upper_threshold)
    return norm_data



def zoom_to_size(image, output_dim):#An implementation of the scipy.ndimage.zoom function.
    input_dim    = n.shape(image)
    if len(input_dim) != len(output_dim):
        print('ERROR [zoom_to_size()]: Cannot change the dimensions of the tensor. The output dims must be the same rank as the input dims.')
        sys.exit()
    output_image = ndi.zoom(image, n.divide(output_dim, n.shape(input_dim)))
    return output_image

def DFR(relativePath):
    print('     DicomFolderRead():')
    dataFiles1 = os.popen('cd ' + relativePath +'; ls').read().split('\n')
    dataFiles = []
    for i in dataFiles1:
        if i!='':
            if (i[len(i)-3:] == 'dcm' or i[len(i)-3:] == 'DCM' or i[len(i)-3:] == 'ima' or i[len(i)-3:] == 'dcm' or i[len(i)-3:] == 'IMA') :
                dataFiles.append(  str(relativePath) + '/' + i)
    return dataFiles

def sum_rebin(image, rebinning_factor):
    print('     sum_rebin():')
    shape = image.shape
    reshape_dims = []
    if len(rebinning_factor) != len(shape):
        print('         You have specified a list of rebinning factors which isnt the same length as the dimensions of the input array. If you want to skip a list, specify a rebinning factor of 1.')
        return
    reshape_dims = []
    sum_axis     = tuple([(-2*i - 1) for i in range(len(rebinning_factor))])
    for i in range(len(rebinning_factor)):
              reshape_dims.append(shape[i]//rebinning_factor[i])
              reshape_dims.append(rebinning_factor[i])
    reshaped_image = image.reshape(reshape_dims).sum(axis = sum_axis)
    return reshaped_image

def mean_rebin(image, rebinning_factor):
    print('     mean_rebin():')
    shape = image.shape
    reshape_dims = []
    if len(rebinning_factor) != len(shape):
        print('You have specified a list of rebinning factors which isnt the same length as the dimensions of the input array. If you want to skip a list, specify a rebinning factor of 1.')
        return
    reshape_dims = []
    sum_axis     = tuple([(-2*i - 1) for i in range(len(rebinning_factor))])
    for i in range(len(rebinning_factor)):
        reshape_dims.append(shape[i]//rebinning_factor[i])
        reshape_dims.append(rebinning_factor[i])
    reshaped_image = image.reshape(reshape_dims).mean(axis = sum_axis)
    return reshaped_image


#A new alternative to pad_to_N() which will not trim/pad slices from a scan that is larger/smaller than 256 by 56, but will interpolate using ndimage.zoom
def transform_to_N(image, N):
    start  = time.time()
    print('     transform_to_N():')
    slices = n.shape(image)[0]
    if slices >= 213 and slices <= 276:
        padded = True
        end    = time.time()
        print('     end transform_to_N() ' + str(n.round(end-start,3)) + ' seconds')
        return pad_to_N(image,N), padded
    else:
        padded      = False
        image_shape = n.shape(image)
        zoom_image  = ndi.zoom(image, n.divide([256,image_shape[1],image_shape[2]], image_shape))
        end         = time.time()
        return zoom_image, padded

def pad_to_N(image, N):
    print('     pad_to_N():')
    slices = n.shape(image)[0]
    if slices == N:
        return image
    elif slices < N:
        print('         Padding to ', N)
        while n.shape(image)[0] != N:
            if is_odd(slices):
                image = n.insert(image, 0, n.zeros(n.shape(image[0])), axis=0)
            else:
                image = n.insert(image, len(image)-1, n.zeros(n.shape(image[0])), axis=0)
    elif slices > N:
        print('         Trimming to ', N)
        while n.shape(image)[0] != N:
            if is_odd(slices):
                image = n.delete(image, 0, axis=0)
            else:
                image = n.delete(image, len(image)-1, axis=0)
    return image

def is_odd(num):
    return bool(num & 0x1)

def get_CT_data(CT_dcm_directory):
    print('    get_data():')
    CT_dicom_files = DFR(CT_dcm_directory)
    CT_data        = []
    z1            = pydicom.read_file(CT_dicom_files[0], force=True).ImagePositionPatient[2]
    z2            = pydicom.read_file(CT_dicom_files[1], force=True).ImagePositionPatient[2]
    print('         collecting CT data...')
    for file in CT_dicom_files:
        CT_data.append(pydicom.read_file(file, force = True).pixel_array)
    CT_data        = n.array(CT_data)
    CT_data        = apply_window(CT_data, upper_threshold = 3000)
    if z1>z2:
        print('         flipping dataset to convention...')
        CT_data = CT_data[::-1]
        Flip    = True
    else:
        Flip    = False
    return CT_data, Flip



def get_CTandNM_data(CT_dcm_directory, NM_dcm_directory, modality = 'PT'):
    start = time.time()
    print('    get_data():')
    CT_dicom_files = DFR(CT_dcm_directory)
    CT_data        = []
    z1             = pydicom.read_file(CT_dicom_files[0], force=True).ImagePositionPatient[2]
    z2             = pydicom.read_file(CT_dicom_files[1], force=True).ImagePositionPatient[2]
    print('         collecting CT data...')
    for file in CT_dicom_files:
        CT_data.append(pydicom.read_file(file, force=True).pixel_array)
    CT_data        = n.array(CT_data)
    NM_data        = []
    NM_dicom_files = DFR(NM_dcm_directory)
    print('         collecting NM data...')
    for file in NM_dicom_files:
        NM_data.append(pydicom.read_file(file, force=True).pixel_array)
    NM_data        = n.array(NM_data)
    if z1 > z2:
        print('         flipping dataset to convention...')
        CT_data = CT_data[::-1]
        Flip    = True
    else:
        Flip    = False
    print('         data collected.')
    end = time.time()
    print('         time elapsed ' + str(end - start) + ' seconds')
    return CT_data, NM_data, Flip

def get_dicom_header_info_CT(CT_dcm_directory):
    print('     get_dicom_header_info():')
    CT_dicom_files = DFR(CT_dcm_directory)
    print('')
    ds_CT          = []
    for ctFile in CT_dicom_files:
        ds_CT.append(pydicom.read_file(ctFile, force=True))
    return ds_CT

def get_dicom_header_info_CTandNM(CT_dcm_directory, NM_dcm_directory):
    CT_dicom_files = DFR(CT_dcm_directory)
    NM_dicom_files = DFR(NM_dcm_directory)
    ds_CT          = []
    ds_NM          = []
    for ctFile in CT_dicom_files:
        ds_CT.append(pydicom.read_file(ctFile, force = True))
    for nmFile in NM_dicom_files:
        ds_NM.append(pydicom.read_file(nmFile, force = True))
    return ds_CT, ds_NM


#Get the start and end index of the organ using the prediction from the localisation neural network. It starts with the max z value and goes across until the slice value decreases to below the specified threshold.
def get_projection_bounds(projection, trim_threshold=0.3, buffer = [5,5]):
    max_index = n.argmax(projection)
    for i in range(len(projection)):
        if projection[i] > trim_threshold * projection[max_index]:
            start_slice = i - buffer[0]
            break
    for i in range(len(projection)):
        if projection[len(projection) - 1 - i] > trim_threshold * projection[max_index]:
            end_slice   = len(projection) - i + buffer[1]
            break
    try:
        print('         Neural network prediction slice range (start, end): ', (start_slice, end_slice))
    except:
        print('\n\n\n ERROR: Could not locate the start and end bounds of the organ from the localisation neural network output. \n\n\n')
    if start_slice <0:
        start_slice = 0
    else:
        pass
    if end_slice > len(projection):
        end_slice = len(projection)
    else:
        pass
    return start_slice, end_slice

def get_ROI_Z_CoM(image, Localisation_Threshold = 0.3, buffer = [[5,5],[5,5],[5,5]]):
    start = time.time()
    print('     get_ROI_Z_CoM()')
    image_xy_proj = n.sum(image, axis=(1,2))
    image_xz_proj = n.sum(image, axis=(0,2))
    image_yz_proj = n.sum(image, axis=(0,1))
    #I was having trouble with a DC component to my localisation projections. You base your bounds on a threshold defined as a fraction of the peak. If the background DC component is alread 0.75 of the peak, you become very dependent on your choice. If we normalise between 0 and 1, our threshold value is more robust.
    image_xy_proj = normalise(image_xy_proj)
    image_xz_proj = normalise(image_xz_proj)
    image_yz_proj = normalise(image_yz_proj)

    start_slice, end_slice = get_projection_bounds(image_xy_proj, trim_threshold = Localisation_Threshold, buffer = buffer[0])
    start_row  , end_row   = get_projection_bounds(image_xz_proj, trim_threshold = Localisation_Threshold, buffer = buffer[1])
    start_col  , end_col   = get_projection_bounds(image_yz_proj, trim_threshold = Localisation_Threshold, buffer = buffer[2])
    crop_coords = [start_slice, end_slice, start_row , end_row , start_col , end_col]

    #perform the cm calculation on the resulting line. Should give close to the middle of the ROI.
    CM_data = []
    for i in range(len(image_xy_proj)):
        CM_data.append(i*image_xy_proj[i])
    CM = n.sum(CM_data)/n.sum(image_xy_proj)
    print('         Neural network prediction slice index centre of mass: ', CM)
    print('         organ slice range collected.')
    end = time.time()
    print('         time elapsed ' + str(end - start) + ' seconds')
    return crop_coords

def tensorflow_format(input_array):
    return n.array([n.moveaxis(n.array([input_array]),0,-1)])

#Changed from mean_rebin to ndi.zoom 20200424 See if this suffices as a way to solve the slice number problem.
def Organ_Localiser_CT(CT_data, Rebinned_CT_Data, Localiser_Dimensions, MODEL, Localisation_Threshold = 0.3, buffer = [[5,5],[5,5],[5,5]]):
    print('     Organ_Localiser()')
    rebin_factor         = n.divide(Localiser_Dimensions, n.shape(CT_data))
    zoom_factor          = n.divide(Localiser_Dimensions, n.shape(Rebinned_CT_Data))
    print('     Zoom factor: ', zoom_factor)
    Localiser_CT_Rebin   = ndi.zoom(Rebinned_CT_Data, zoom_factor)
    Localiser_CT_Rebin   = normalise(Localiser_CT_Rebin)
    Localiser_Prediction = MODEL.predict(tensorflow_format(Localiser_CT_Rebin), batch_size = 1)
    crop_coords      = get_ROI_Z_CoM(Localiser_Prediction[0,:,:,:,0], Localisation_Threshold = Localisation_Threshold, buffer = buffer)
    print('         organ localisation completed.')
    print('         rescaling to original image.')
    rescale_crop_coords = n.array([int(crop_coords[i]*(1/rebin_factor[int(i/2)])) for i in range(len(crop_coords))])
    print('         rescaled organ slice coordinates : ', rescale_crop_coords)
    return rescale_crop_coords


def Organ_Localiser_CTandNM(CT_data, NM_data, CT_rebin_dimensions, MODEL, Localisation_Threshold = 0.3):
    start = time.time()
    print('     Organ_Localiser()')
    Localiser_CT_Rebin   = mean_rebin(CT_data, CT_rebin_dimensions)
    Localiser_NM_Rebin   = sum_rebin(NM_data, [CT_rebin_dimensions[0],1,1])
    Localiser_Prediction = MODEL.predict(tensorflow_format(Localiser_CT_Rebin), batch_size = 1)
    crop_coords      = get_ROI_Z_CoM(Localiser_Prediction[0,:,:,:,0], Localisation_Threshold = Localisation_Threshold)
    print('         organ localisation completed.')
    print('         rescaling to original image.')
    rescale_crop_coords = n.array([crop_coords[i]*CT_rebin_dimensions[int(i/2)] for i in range(len(crop_coords))])
    print('         rescaled organ slice coordinates : ', rescale_crop_coords)
    end = time.time()
    print('         time elapsed ' + str(end - start) + ' seconds')
    return rescale_crop_coords

def generate_NM_Crop_Coordinates(CT_Crop_Coords, CT_h, NM_h):
    CT_voxels       = n.array([1, float(CT_h.Rows), float(CT_h.Columns)])
    NM_voxels       = n.array([1, float(NM_h.Rows), float(NM_h.Columns)])
    rescale_factors = n.divide(NM_voxels, CT_voxels)
    NM_Crop_Coords  = n.array([int(n.round(CT_Crop_Coords[i] * rescale_factors[int(i/2)])) for i in range(len(CT_Crop_Coords))])
    return NM_Crop_Coords

def fit_to_template_3D(image, template):
    print('     fit_to_template_3D()')
    zoom_factors = n.divide(n.shape(template), n.shape(image))
    zoom_image   = ndi.zoom(image, zoom = zoom_factors)
    return zoom_image

def normalise(image):
    image = n.subtract(image, n.amin(image))
    return n.divide(image, n.amax(image))

def thresholdPrediction(i,threshold):
    if i < threshold:
        return 0
    else:
        return 1
_thresholdPrediction = n.vectorize(thresholdPrediction)

