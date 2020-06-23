#!/usr/bin/env python3
#
# A class for parsing the Settings.prf file upon starting the model.
#
# cdp 20200417
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
import xml.etree.ElementTree as ET
start = time.time()

class SettingsParser :
    def __init__(self, settings_file_path):
        settings_file          = ET.parse(settings_file_path)
        settings               = settings_file.getroot()
        self.version           = settings.find('version').text
        self.temp_directory    = settings.find('temp').text
        self.model_directory   = settings.find('ModelRootDir').text
        self.RTStruct_template = settings.find('RTStructureTemplate').text
        loc_models             = settings[4]
        network                = settings[5]
        #Create catching arrays for localised models.
        localisation_models    = []
        localisation_threshold = []
        localisation_buffer    = []
        localisation_shape     = []
        segmentation_models    = []
        segmentation_shape     = []
        segmentation_threshold = []
        roi_colour             = []
        roi_name               = []
        roi_description        = []
        smoothing_factor       = []
        # Create catching array for the network details of all the locations you want to send data to.
        send_back_networks     = []
        #Create catching array for the segmentation model network details.
        segment_node_network   = []
        #Collect models
        for loc_model in loc_models.findall('Model'):
            smoothing_factor.append(float(loc_model.find('SmoothingFactor').text))
            localisation_models.append(loc_model.find('LocalisationName').text)
            localisation_threshold.append(float(loc_model.find('LocalisationThreshold').text))
            localisation_shape.append(loc_model.find('LocalisationShape').text)
            localisation_buffer.append(loc_model.find('LocalisationBuffer').text)
            segmentation_models.append(loc_model.find('SegmentationName').text)
            segmentation_shape.append(loc_model.find('SegmentationShape').text)
            segmentation_threshold.append(float(loc_model.find('SegmentationThreshold').text))
            roi_colour.append(loc_model.find('ROIColour').text)
            roi_name.append(loc_model.find('ROIName').text)
            roi_description.append(loc_model.find('ROIDescription').text)
        #Collect the segmentation node network details.
        for net in network.findall('SegmentationNode'):
            for detail in net:
                #Try and convert to an int to convert the port from a string to an int. If it doesnt work, then it isnt the port so add it as a string.
                try:    segment_node_network.append(int(detail.text))
                except: segment_node_network.append(detail.text)
        #Collect the network details of the locations you want to send data to.
        for net in network.findall('ReceivingEntities'):
            #pick out entities to send to
            for sendTo in net:
                entity = []
                #pick out the details for each entity.
                for network_detail in sendTo:
                    try:
                        entity.append(int(network_detail.text))
                    except:
                        entity.append(network_detail.text)
                send_back_networks.append(entity)
        #items which need to be converted to lists:
        #Note that you should not edit these in the code anywhere. The whole point of having a settings file is for easy editing of the input parameters.
        self.localisation_buffer                 = self.convert_string_arrays_to_int(localisation_buffer)
        self.localisation_shape                  = self.convert_string_arrays_to_int(localisation_shape)
        self.segmentation_shape                  = self.convert_string_arrays_to_int(segmentation_shape)
        self.roi_colour                          = self.convert_string_arrays_to_int(roi_colour)
        self.localisation_models                 = localisation_models
        self.segmentation_models                 = segmentation_models
        self.localisation_threshold              = localisation_threshold
        self.segmentation_threshold              = segmentation_threshold
        self.segmentation_server_network_details = segment_node_network
        self.send_network_details                = send_back_networks
        self.roi_name                            = roi_name
        self.roi_description                     = roi_description
        self.smoothing_factor                    = smoothing_factor

    def convert_string_arrays_to_int(self,string_array):
        if string_array[0][0] == '[':
            return_array = []
            for i in string_array:
                model_array = []
                couple1 = []
                couple2 = []
                couple3 = []
                i1 = i.replace('[','')
                i2 = i1.replace(']','')
                comma_split = i2.split(',')
                for i in range(2):
                    couple1.append(int(comma_split[i]))
                    couple2.append(int(comma_split[i+2]))
                    couple3.append(int(comma_split[i+4]))
                model_array.append(couple1)
                model_array.append(couple2)
                model_array.append(couple3)
                return_array.append(model_array)
            return return_array
        else:
            return_array = []
            for i in string_array:
                triple = []
                comma_split = i.split(',')
                for char in comma_split:
                    triple.append(int(char))
                return_array.append(triple)
            return return_array

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_file', type = str, help = 'The preferences file you wish to parse.')
    args = parser.parse_args()
    SettingsParser(args.xml_file)