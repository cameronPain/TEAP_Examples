#!/usr/bin/env python3
#
#
# The architecture of this code:
# On C_Store event:
#   Save the incomming dicom file to a local directory
#
# On connection closed:
#   We have the data and the connection is closed. We can load and do the processing within the OnConnectionClosed function. Note we cannot os.popen into another window as the NeuralNetworks need to be ran in the __main__ instance. Once this is ran, the server should go back to a state of listening.
#
# cdp 20200404
#
# cdp 20200405 I upgraded to the developer version of pynetdicom (1.5.0.dev0) as this has the ability to include additional args into the callback functions. This will allow me to define a NeuNetProcessing class and have access to the information sent between AETitles in the association. The previous verison pynetdicom 1.4.1 did not have this. The server runs on a different thread, so you would need to load your tensorflow models each time an association is made which would take an additional 60 seconds so I want to avoid this.
#
#
#
from pydicom import dcmread, FileDataset, Dataset
from pydicom.uid import ImplicitVRLittleEndian, JPEGBaseline
from pynetdicom import AE, evt, StoragePresentationContexts, association, debug_logger, build_context
from pynetdicom.sop_class import NuclearMedicineImageStorage, CTImageStorage, RTStructureSetStorage, VerificationSOPClass, SecondaryCaptureImageStorage
from pynetdicom.association import Association
import pynetdicom
pnd_version = pynetdicom.__version__.split('.')
if int(pnd_version[0]) < 1 or int(pnd_version[1]) < 5:
    print('ERROR: Running pynetdicom.__version__ = ' + pynetdicom.__version__ + ' Upgrade to 1.5.0.dev0 \n pip install git+https://github.com/pydicom/pynetdicom.git')
    exit()
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
workingDir = os.popen('pwd').read().split('\n')[0]
sys.path.append(workingDir + '/Imports/')
print(workingDir + '/Imports/' + ' appended to sys.path')
start = time.time()
print('     removing FutureWarning prints...')
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow
import tensorflow.keras.models
#local imports
from fileViewer_cdp       import fileViewer
from buildRTStructure_cdp import RT_Structure
from SegmentationProcess_cdp import Segmentation_Process
from DicomServer_cdp import DicomServer
from SettingsParser_cdp import SettingsParser


class NeuNetSeg: #The class that ties everything together.
    def __init__(self, settings):
        self.settings_file        = settings
        self.settings             = SettingsParser(settings)
        self.dicom_server         = DicomServer(*self.settings.segmentation_server_network_details, self.settings.temp_directory)
        self.segmentation_process = Segmentation_Process()
        self.RT_Structure         = RT_Structure(self.settings.RTStruct_template)
        #etc.

    def refresh_segmentation_algorithm(self):
        self.settings             = SettingsParser(self.settings_file)
        self.segmentation_process = Segmentation_Process()
        self.RT_Structure         = RT_Structure(self.settings.RTStruct_template)
        os.popen('rm -rf ' + self.settings.temp_directory)
        time.sleep(0.1)
        os.popen('mkdir ' + self.settings.temp_directory)
        time.sleep(0.1)

    def initialise_models(self):
        print('\n\n\n################################## INITIALISING SEGMENTATION ALGORITHM ##################################')
        print('-------------------------------------------------------------------------------')
        print('NeuNetSeg' + self.settings.version)
        print('Working directory   : ' + self.settings.temp_directory)
        print('Model directory     : ' + self.settings.model_directory)
        print('RTStructure template: ' + self.settings.RTStruct_template)
        print('--------------------------------------------------------------------------------')
        print('Segmentation node network details:')
        print('--------------------------------------------------------------------------------')
        print('     AE Title: ', self.settings.segmentation_server_network_details[0])
        print('           ip: ', self.settings.segmentation_server_network_details[1])
        print('         port: ', self.settings.segmentation_server_network_details[2])
        print('--------------------------------------------------------------------------------')
        print('Sending data back to the following network locations:')
        print('--------------------------------------------------------------------------------')
        for network_location in self.settings.send_network_details:
            print('     AE Title: ', network_location[0])
            print('           ip: ', network_location[1])
            print('         port: ', network_location[2])
            print('--------------------------------------------------------------------------------')
        self.localisation_models            = []
        self.segmentation_models            = []
        print('Loading localisation models:')
        init_start = time.time()
        start      = time.time()
        for loc_model in self.settings.localisation_models:
            print('     ' + loc_model)
            self.localisation_models.append(tensorflow.keras.models.load_model(self.settings.model_directory + loc_model))
        end        = time.time()
        print('Localisation models loaded (took ', n.round(end-start,6), ' seconds, total of ', n.round(end - init_start, 6), ' seconds)')
        print('--------------------------------------------------------------------------------')
        print('Loading segmentation models:')
        start      = time.time()
        for seg_model in self.settings.segmentation_models:
            print('     ' + seg_model)
            self.segmentation_models.append(tensorflow.keras.models.load_model(self.settings.model_directory + seg_model))
        end        = time.time()
        print('Segmentation models loaded (', n.round(end-start, 6), ' seconds, total of ', n.round(end - init_start, 6), ' seconds)')
        print('--------------------------------------------------------------------------------')
        start      = time.time()
        print('Initialising localisation models:')
        for i in range(len(self.localisation_models)):
            print('     Initialising...', end = '\r')
            init_im = n.array([n.moveaxis(n.array([n.zeros(self.settings.localisation_shape[i])]),0,-1)])
            preditction = self.localisation_models[i].predict(init_im)
            print('     Done...', end = '\r')
        end        = time.time()
        print('Localisation models initialised (', n.round(end - start,6), ' seconds, total of ', n.round(end - init_start, 6), ' seconds)')
        print('--------------------------------------------------------------------------------')
        print('Initialising segmentation models:')
        for i in range(len(self.segmentation_models)):
            print('     Initialising...', end = '\r')
            init_im = n.array([n.moveaxis(n.array([n.zeros(self.settings.segmentation_shape[i])]),0,-1)])
            preditction = self.segmentation_models[i].predict(init_im)
            print('     Done...', end = '\r')
        end        = time.time()
        print('Segmentation models initialised (', n.round(end - start,6), ' seconds, total of ', n.round(end - init_start, 6), ' seconds)')
        print('--------------------------------------------------------------------------------')
        self.ROIGenerationAlgorithm         = 'Cameron Pain: Convolutional Neural Networks'
        print('################################## ALGORITHM INITIALISED ##################################\n\n\n')

class server_segmentation_mediator :
    def __init__(self):
        self.CSTORE_Modality     = False
        self.isFirstConnection   = True
        self.save_dir            = None
        self.current_study       = None
        self.current_series      = None
        self.current_working_dir = None

def OnReceiveStore(event, Running_AE, com): #Save the incoming dicom into a temp folder.
    #print('Data received.')
    ds           = event.dataset
    ds.file_meta = event.file_meta
    Modality     = ds.Modality
    if Modality != 'CT':
        com.CSTORE_Modality = False #The modality received in the C_STORE event was not supported.
        print('This segmentation algorithm accepts Whole body CT data only. The dicom tag "Modality" returned a value that was not "CT".')
        return 0x0000
    else:
        com.CSTORE_Modality = True #The modality received in the C_STORE event is supported.
    #print(com.CSTORE_Modality)
    if com.isFirstConnection:
        com.current_study     = ds.StudyInstanceUID[-6:]
        com.current_series    = ds.SeriesInstanceUID[-6:]
        study_dir             = ds.StudyInstanceUID + '/'
        series_dir            = ds.SeriesInstanceUID + '/'
        os.popen('mkdir ' + Running_AE.temp + study_dir + ';' + 'mkdir ' + Running_AE.temp + study_dir + series_dir)
        com.isFirstConnection = False
        com.save_dir          = Running_AE.temp + study_dir + series_dir
    elif ds.SeriesInstanceUID[-6:] != com.current_series:
        com.current_study     = ds.StudyInstanceUID[-6:]
        com.current_series    = ds.SeriesInstanceUID[-6:]
        study_dir             = ds.StudyInstanceUID + '/'
        series_dir            = ds.SeriesInstanceUID + '/'
        os.popen('mkdir ' + Running_AE.temp + study_dir + ';' + 'mkdir ' + Running_AE.temp + study_dir + series_dir)
        com.isFirstConnection = False
        com.save_dir          = Running_AE.temp + study_dir + series_dir
    print(' File UIDs:')
    print('     SOP Class UID          : ' + ds.SOPClassUID)
    print('     SOP Instance UID       : ' + ds.SOPInstanceUID)
    print('     Study Instance UID     : ' + ds.StudyInstanceUID)
    print('     Series Instance UID    : ' + ds.SeriesInstanceUID)
    print('     Frame of Reference UID : ' + ds.FrameOfReferenceUID)
    instance_no       = str(ds.InstanceNumber)
    desired_character_length = 10
    file_prefix = 'IM_'
    for i in range(desired_character_length - len(instance_no)):
        file_prefix = file_prefix + '0'
    modality  = str(ds.Modality)
    save_file = com.save_dir + file_prefix + instance_no + '.dcm'
    ds.save_as(save_file)
    return 0x0000   #Return the hexadecimal 0 which is interpereted as success in DICOM talk.

def OnConnectionClosed(event, Running_AE, com):
    com.isFirstConnection = True
    print(com.CSTORE_Modality)
    if com.CSTORE_Modality:
        Running_AE.shutdown()
        return 0x0000
    else:
        print('Connection closed.')
        print('AE running...')
        return 0x0000

def main(settings_file):
    neunetseg = NeuNetSeg(settings_file)
    neunetseg.initialise_models()
    com        = server_segmentation_mediator()
    neunetseg.dicom_server.specify_dicom_communication_function(evt.EVT_C_STORE, OnReceiveStore, [neunetseg.dicom_server, com])
    neunetseg.dicom_server.specify_dicom_communication_function(evt.EVT_CONN_CLOSE, OnConnectionClosed, [neunetseg.dicom_server, com])
    while True:
        #Define an error value to determine which part of the code an error occured in
        #error_code = -1
        try: #If you pick up an error during the processing, return to the initial state.
            print('--------------------------------------------------------------------------------')
            print('Server started')
            print('--------------------------------------------------------------------------------')
            neunetseg.dicom_server.start_server(neunetseg)
            start_timer = time.time()
            print('--------------------------------------------------------------------------------')
            print('Processing input data')
            print('--------------------------------------------------------------------------------')
            neunetseg.segmentation_process.ProcessDataForInput(com.save_dir)
            print('--------------------------------------------------------------------------------')
            print('Connecting RT structure to the delivered dicom series.')
            print('--------------------------------------------------------------------------------')
            neunetseg.RT_Structure.update_RTStructurePatientDetails(neunetseg)
            neunetseg.RT_Structure.update_ReferencedFrameOfReferenceSequence(neunetseg)
            print('--------------------------------------------------------------------------------')
            print('Starting the segmentation process.')
            print('--------------------------------------------------------------------------------')
            neunetseg.segmentation_process.PerformNeuNetSegmentation(neunetseg)
            print('--------------------------------------------------------------------------------')
            print('Saving the segmentation into the connected RT structure.')
            print('--------------------------------------------------------------------------------')
            neunetseg.RT_Structure.update_StructureSetROISequence_and_ROIContourSequence(neunetseg)
            print('--------------------------------------------------------------------------------')
            print('Saving the RT structure locally.')
            print('--------------------------------------------------------------------------------')
            neunetseg.RT_Structure.RTStruct.save_as( com.save_dir + 'Segmentation_RTStructure.dcm')
            print('--------------------------------------------------------------------------------')
            print('Sending RT structures to the return application entities.')
            print('--------------------------------------------------------------------------------')
            for i in range(len(neunetseg.settings.send_network_details)):
                neunetseg.dicom_server.send(com.save_dir + 'Segmentation_RTStructure.dcm', *neunetseg.settings.send_network_details[i])
            print('--------------------------------------------------------------------------------')
            print('Refreshing the segmentation algorithm.')
            print('--------------------------------------------------------------------------------')
            neunetseg.refresh_segmentation_algorithm()
            end_timer = time.time()
            print('--------------------------------------------------------------------------------')
            print('Single segmentation took ', end_timer - start_timer, ' seconds.')
            print('--------------------------------------------------------------------------------')
        except:
            print('ERROR: Reverting to initial state.')
            neunetseg.refresh_segmentation_algorithm()

import argparse
if __name__ == '__main__':
    usage  = 'Cameron Pain (cameron.pain@austin.org.au): Python3 dicom node for neural network segmentation of WBCT. Listens on a specified port at a specified IP address. The details of the models and dicom locations to send to are in the Settings.xml file. If you want to edit the Settings.xml file, make a copy first.'
    parser = argparse.ArgumentParser(description = usage)
    parser.add_argument('--settings_file', dest = 'settings_file', default = 'settings.xml', type = str, help = 'You may specify the path to a specific settings file. The default is a file named settings.xml located in the same directory as the main() function.')
    args = parser.parse_args()
    main(args.settings_file)










