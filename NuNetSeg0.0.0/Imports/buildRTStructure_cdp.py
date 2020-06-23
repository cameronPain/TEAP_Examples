#!/usr/bin/env python3
#
# Define a number of functions which are used to update a RTStructure template with contours specifed in a contour library of the form:
#           [ [], [], [ [0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,], [2,2,0,3,2,0,3,3,0,2,3,0,2,2,0]], [] ]
#             ^                     ^                                   ^
#      no contour slice 0      contour 0 slice 2             contour 1 slice 2
#
# Updates the RT Structure with patient details from a specified Dicom series. Most likely a CT which you want to define the RT structure to
#
# Redefines the necessary UIDs to those from a corresponding Dicom series to associate the RT structure to said series.
#
# cdp 20200305
#
import pydicom
import numpy as n
import matplotlib.pyplot as pyplot
import sys
import pydicom
from DicomFolderRead import DicomFolderRead as DFR
import datetime

class RT_Structure:

    def __init__(self, template):
        self.RTStruct = pydicom.read_file(template, force = True)

    def update_ReferencedFrameOfReferenceSequence(self, NeuNetSeg): #void  #referenced CT dicom stack is an iterable containing the pydicom.DicomObjects of each slice. The UIDs from each slice are needed for the RT Structure.
        referenced_dicom_stack = NeuNetSeg.segmentation_process.Dicom_Header
        """
        #####################      STRUCTURE OF THE REFERENCED_FRAME_OF_REFERENCE_SEQUENCE TAG      #######################
        ReferencedFrameOfReferenceSequence   ----->   pydicom.sequence.Sequence
                        |
                        v
                        item                        -----> pydicom.dataset.Dataset
                        |
                        v
                        FrameOfReferenceUID               -----> pydicom.uid.UID
                        RTReferencedStudySequence         -----> pydicom.sequence.Sequence
                                    |
                                    v
                                    item                            -----> pydicom.dataset.Dataset
                                    |
                                    v
                                    ReferencedSOPClassUID                 -----> pydicom.uid.UID
                                    ReferencedSOPInstanceUID              -----> pydicom.uid.UID
                                    RTReferencedSeriesSequence            -----> pydicom.sequence.Sequence
                                                |
                                                v
                                                item                               -----> pydicom.dataset.Dataset
                                                |
                                                v
                                                SeriesInstanceUID                        -----> pydicom.uid.UID
                                                ContourImageSequence                     -----> pydicom.sequence.Sequence
                                                            |
                                                            v
                                                            item                                    -----> pydicom.dataset.Dataset
                                                            item                                    -----> pydicom.dataset.Dataset
                                                            .                                                          .
                                                            .                                                          .
                                                            .                                                          .
                                                            .                                                          .
                                                            item                                    -----> pydicom.dataset.Dataset
                                                            |
                                                            V
                                                            ReferencedSOPClassUID                         -----> pydicom.uid.UID
                                                            ReferencedSOPInstanceUID                      -----> pydicom.uid.UID
                                                          
        Build a template of this structure and paste in the relevant study information. Start from the most nested datasets and work outwards.
        """
        #ContourImageSequence.
        #Collect the datasets containing UIDs for each referenced slice.
        ImageStorageUID                 = referenced_dicom_stack[0].SOPClassUID #UID for the image type used.
        DetachedStudyManagementSOPClass = '1.2.840.10008.3.1.2.3.1'   #UID for DetachedStudyManagementSOPClass
        nof_slices                      = len(referenced_dicom_stack)
        ContourImageSequence_iter       = []
        #Build an iterable containing a dataset for each slice file.
        for i in range(nof_slices):
            item_dataset = pydicom.dataset.Dataset()
            item_dataset.ReferencedSOPClassUID    = referenced_dicom_stack[i].SOPClassUID
            item_dataset.ReferencedSOPInstanceUID = referenced_dicom_stack[i].SOPInstanceUID
            ContourImageSequence_iter.append(item_dataset)
        #Create a dicom sequence and hand it the iterable.
        #ContourImageSequence
        ContourImageSequence   = pydicom.sequence.Sequence(ContourImageSequence_iter)
            #Create Series Instance UID
        SeriesInstanceUID      = referenced_dicom_stack[0].SeriesInstanceUID
        #put into RTReferencedSeriesSequenceDataset
        RTReferencedSeriesSequenceDataset = pydicom.dataset.Dataset()
        RTReferencedSeriesSequenceDataset.ContourImageSequence = ContourImageSequence
        RTReferencedSeriesSequenceDataset.SeriesInstanceUID    = SeriesInstanceUID
        #Paste this dataset into the RTReferencedSeriesSequence sequence.
        RTReferencedSeriesSequence        = pydicom.sequence.Sequence([RTReferencedSeriesSequenceDataset])
        #Create UIDs
        ReferencedSOPInstanceUID          = referenced_dicom_stack[len(referenced_dicom_stack)-1].SOPInstanceUID
        ReferencedSOPClassUID             = pydicom.uid.UID(DetachedStudyManagementSOPClass)
        #Paste this layer into a dataset.
        RTReferencedStudySequenceDataset  = pydicom.dataset.Dataset()
        RTReferencedStudySequenceDataset.ReferencedSOPClassUID      = ReferencedSOPClassUID
        RTReferencedStudySequenceDataset.ReferencedSOPInstanceUID   = ReferencedSOPInstanceUID
        RTReferencedStudySequenceDataset.RTReferencedSeriesSequence = RTReferencedSeriesSequence
        #Paste into a sequence.
        RTReferencedStudySequence         = pydicom.sequence.Sequence([RTReferencedStudySequenceDataset])
        #create FrameOReferenceUID
        FrameOfReferenceUID               = referenced_dicom_stack[0].FrameOfReferenceUID
        #create ReferencedFrameOfReferenceSequence dataset
        ReferencedFrameOfReferenceSequenceDataset                           = pydicom.dataset.Dataset()
        ReferencedFrameOfReferenceSequenceDataset.FrameOfReferenceUID       = FrameOfReferenceUID
        ReferencedFrameOfReferenceSequenceDataset.RTReferencedStudySequence = RTReferencedStudySequence
        #Finally, create the ReferencedFrameOfReferenceSequence to be pasted into the template file.
        ReferencedFrameOfReferenceSequence = pydicom.sequence.Sequence([ReferencedFrameOfReferenceSequenceDataset])
        self.RTStruct.ReferencedFrameOfReferenceSequence = ReferencedFrameOfReferenceSequence
        #end update_ReferencedFrameOfReferenceSequence

    def update_StructureSetROISequence_and_ROIContourSequence(self, NeuNetSeg): #void
        print('     Updating the StructureSetROISequence tag...')
        referenced_dicom_stack = NeuNetSeg.segmentation_process.Dicom_Header
        ROIGenerationAlgorithm = NeuNetSeg.ROIGenerationAlgorithm
        for i in range(len(NeuNetSeg.segmentation_process.Dicom_Contour_Library)):
            contour_library        = NeuNetSeg.segmentation_process.Dicom_Contour_Library[i]
            ROIName                = NeuNetSeg.settings.roi_name[i]
            ROIDescription         = NeuNetSeg.settings.roi_description[i]
            ROIColour              = NeuNetSeg.settings.roi_colour[i]
            """
                #####################      STRUCTURE OF THE STURCTURE_SET_ROI_SEQUENCE TAG     #######################
                StructureSetROISequence         -----> pydicom.sequence.Sequence
                |
                v
                item                          -----> pydicom.dataset.Dataset
                |
                v
                ROINumber                           -----> pydicom.valuerep.IS
                ReferencedFrameOfReferenceUID       -----> pydicom.uid.UID
                ROIName                             -----> str
                ROIDescription                      -----> str
                ROIGenerationAlgorithm              -----> str
                
                Build the new StructureSetROISequence from the deepest layer outwards.
                """
            #Want to keep contours that are already in the file and just add our new one onto it. This way we can use this file to add our liver, kidney, lungs, spine etc as we see fit.
            StructureSetROISequenceDataset_Iter = []
            try:
                for item in self.RTStruct.StructureSetROISequence:
                    if item != pydicom.dataset.Dataset():
                        StructureSetROISequenceDataset_Iter.append(item)
                    else:
                        pass
            except:
                pass
            new_ROI                               = pydicom.dataset.Dataset()
            new_ROI.ROINumber                     = pydicom.valuerep.IS(len(StructureSetROISequenceDataset_Iter) + 1)
            new_ROI.ROIName                       = ROIName
            new_ROI.ROIDescription                = ROIDescription
            new_ROI.ROIGenerationAlgorithm        = ROIGenerationAlgorithm
            new_ROI.ReferencedFrameOfReferenceUID = referenced_dicom_stack[0].FrameOfReferenceUID
            print('\n\n\n')
            print('New ROI number: ', new_ROI.ROINumber)
            print('Source of ROI number: ', StructureSetROISequenceDataset_Iter)
            StructureSetROISequenceDataset_Iter.append(new_ROI)
            #Put our new iterable of datasets into a sequence.
            StructureSetROISequence               = pydicom.sequence.Sequence(StructureSetROISequenceDataset_Iter)
            #Put the new sequence into the RT structure in place of the old one.
            self.RTStruct.StructureSetROISequence   = StructureSetROISequence
            print('     StuctureSetROISequence updated.')
            #end updated StructureSetROISequence
            print('     Updating the ROIContourSequence...')
            """
                #####################      STRUCTURE OF THE ROI_CONTOUR_SEQUENCE TAG     #######################
                ROIContourSequence      -----> pydicom.sequence.Sequence
                |
                v
                item                 -----> pydicom.dataset.Dataset
                item                 -----> pydicom.dataset.Dataset
                .                    .
                .                    .
                .                    .
                item                 -----> pydicom.dataset.Dataset
                |
                v
                ROIDisplayColor          -----> pydicom.multival.MultiValue
                |
                v
                value                   -----> pydicom.valuerep.IS
                value                   -----> pydicom.valuerep.IS
                value                   -----> pydicom.valuerep.IS
                ContourSequence               -----> pydicom.sequence.Sequence
                |
                v
                item                         -----> pydicom.dataset.Dataset
                item                         -----> pydicom.dataset.Dataset
                .                            .
                .                            .
                .                            .
                item                         -----> pydicom.dataset.Dataset
                |
                v
                ContourImageSequence            -----> pydicom.sequence.Sequence
                |
                v
                item                            -----> pydicom.dataset.Dataset
                |
                v
                ReferencedSOPClassUID              -----> pydicom.uid.UID
                ReferencedSOPInstanceUID           -----> pydicom.uid.UID
                ContourGeometricType                  -----> str
                NumberOfContourPoints                 -----> pydicom.valuerep.IS
                ContourData                           -----> pydicom.multival.MultiValue
                |
                v
                value                                -----> pydicom.valuerep.DSfloat
                value                                -----> pyidcom.valuerep.DSfloat
                .                                    .
                .                                    .
                .                                    .
                value                                -----> pydicom.valuerep.DSfloat
                ReferencedROINumber           -----> pydicom.valuerep.IS
                
                Build the new StructureSetROISequence from the deepest layer outwards.
                """
            #Build the ROI Display colour
            ROIContourSequenceDataset_iter  = []
            try: # if RTStructure.ROIContourSequence is defined, we can perform this to catch previously defined ROIs.
                for item in self.RTStruct.ROIContourSequence:
                    if item != pydicom.dataset.Dataset():
                        print('found unempty entry')
                        ROIContourSequenceDataset_iter.append(item)
                    else:
                        print('Found empty entry')
                        pass
            except: #If this is the first ROI to put into the RT Structure, then the for item in RTStructure.ROIContourSequence will be calling a non-existant tag.
                pass
            ROIDisplayColourMultiValue_iter = ROIColour #Make the ROI white. Add something later which chooses a colour from a library of unused colours.
            ROIDisplayColour                = pydicom.multival.MultiValue(int, ROIDisplayColourMultiValue_iter)
            #Each ROI is contained in the ROIContourSequence. In this code, we put a single ROI in at a time. We will need a for loop to create an item in ContourSequence for each slice containing a contour.
            ContourSequenceDataset_iter = []
            for i in range(len(contour_library)): #scan over each slice contained in the contour library.
                if contour_library[i] == []:
                    continue
                for j in range(len(contour_library[i])): #scan over each contour contained in the slice
                    ContourSequenceDataset  = pydicom.dataset.Dataset()
                    #Create the ContourImageSequence
                    ContourImageSequenceDataset = pydicom.dataset.Dataset()
                    ContourImageSequenceDataset.ReferencedSOPClassUID    = referenced_dicom_stack[0].SOPClassUID
                    ContourImageSequenceDataset.ReferencedSOPInstanceUID = referenced_dicom_stack[i].SOPInstanceUID
                    ContourImageSequence                                 = pydicom.sequence.Sequence([ContourImageSequenceDataset])
                    #Contour geometric type
                    ContourGeometricType                                 = 'CLOSED_PLANAR'
                    #Contour Points
                    NumberOfContourPoints                                = len(contour_library[i][j][0])
                    if NumberOfContourPoints < 9: #We need a minimum of three points to define a closed contour (ie. 3*3 individual coords.) The way we define our contour is at the centre of pixels, so some with just two points will be created.
                        continue
                    ContourDataMultiValue_iter                           = []
                    #iterate over the points in the contour pasting them into the MultiValue_iterable.
                    #contour_library = [ [], [], [], [], [[n.array([1,2,3,4,5,6])],[n.array([5,2,3,4,-2,3,4,2])]], []...] so I need to pick out points in contour_library[i][j][0].
                    for point in contour_library[i][j][0]:
                        ContourDataMultiValue_iter.append(point)
                    #paste the contour points iterable into the MultiValue object.
                    ContourData                                          = pydicom.multival.MultiValue(float, ContourDataMultiValue_iter)
                    #Paste all the data into the Contour Sequence Dataset.
                    ContourSequenceDataset.ContourImageSequence          = ContourImageSequence
                    ContourSequenceDataset.ContourGeometricType          = ContourGeometricType
                    ContourSequenceDataset.NumberOfContourPoints         = NumberOfContourPoints
                    ContourSequenceDataset.ContourData                   = ContourData
                    ContourSequenceDataset_iter.append(ContourSequenceDataset)
            # end for
            # end for
            ContourSequence                                          = pydicom.sequence.Sequence(ContourSequenceDataset_iter)
            #Point to the ROI number in the StructureSetROISequence tag which the contours belong to.
            ReferencedROINumber                                      = new_ROI.ROINumber
            #Build the last layer, ROIContourSequence
            ROIContourSequenceDataset                                = pydicom.dataset.Dataset()
            ROIContourSequenceDataset.ReferencedROINumber            = ReferencedROINumber
            ROIContourSequenceDataset.ContourSequence                = ContourSequence
            ROIContourSequenceDataset.ROIDisplayColor               = ROIDisplayColour
            ROIContourSequenceDataset_iter.append(ROIContourSequenceDataset)
            ROIContourSequence                                       = pydicom.sequence.Sequence(ROIContourSequenceDataset_iter)
            #Paste the constructed ROIContourSequence into the RT Structure.
            self.RTStruct.ROIContourSequence                           = ROIContourSequence

    def update_RTStructure_UIDs(self): #void.
        self.RTStruct.SOPInstanceUID    = pydicom.uid.generate_uid()
        self.RTStruct.SeriesInstanceUID = pydicom.uid.generate_uid()

    def update_RTStructurePatientDetails(self, NeuNetSeg):#void #Paste in patient details from the complementary CT dataset.
        #Simply paste in the details from the referenced dataset into the RTStructure.
        referenced_dicom_stack                           = NeuNetSeg.segmentation_process.Dicom_Header
        self.RTStruct.SOPInstanceUID                     = pydicom.uid.generate_uid()
        self.RTStruct.InstanceCreationDate               = datetime.datetime.now().strftime('%Y%m%d')
        self.RTStruct.InstanceCreationTime               = datetime.datetime.now().strftime('%H%M%S')
        self.RTStruct.StudyDate                          = referenced_dicom_stack[0].StudyDate
        self.RTStruct.SeriesDate                         = referenced_dicom_stack[0].SeriesDate
        self.RTStruct.StudyTime                          = referenced_dicom_stack[0].StudyTime
        self.RTStruct.SeriesTime                         = referenced_dicom_stack[0].SeriesTime
        self.RTStruct.AccessionNumber                    = referenced_dicom_stack[0].AccessionNumber
        self.RTStruct.Manufacturer                       = 'CameronPain'
        self.RTStruct.ReferringPhysicianName             = referenced_dicom_stack[0].ReferringPhysicianName
        self.RTStruct.StationName                        = 'CameronPain: Neural Networks'
        self.RTStruct.StudyDescription                   = referenced_dicom_stack[0].StudyDescription
        self.RTStruct.SeriesDescription                  = referenced_dicom_stack[0].SeriesDescription + '_NeuralNetworkSegmentation'
        self.RTStruct.ManufacturerModelName              = 'CDP_NeuralNetworks_ver0.0'
        self.RTStruct.PatientName                        = referenced_dicom_stack[0].PatientName
        self.RTStruct.PatientID                          = referenced_dicom_stack[0].PatientID
        self.RTStruct.PatientBirthDate                   = referenced_dicom_stack[0].PatientBirthDate
        self.RTStruct.PatientSex                         = referenced_dicom_stack[0].PatientSex
        self.RTStruct.PatientAge                         = referenced_dicom_stack[0].PatientAge
        self.RTStruct.PatientSize                        = referenced_dicom_stack[0].PatientSize
        self.RTStruct.PatientWeight                      = referenced_dicom_stack[0].PatientWeight
        self.RTStruct.SoftwareVersions                   = '0.0'
        self.RTStruct.StudyInstanceUID                   = referenced_dicom_stack[0].StudyInstanceUID
        self.RTStruct.SeriesInstanceUID                  = pydicom.uid.generate_uid()
        self.RTStruct.StudyID                            = referenced_dicom_stack[0].StudyID
        self.RTStruct.SeriesNumber                       = '0'
        self.RTStruct.StructureSetLabel                  = 'RTstruct'
        self.RTStruct.StructureSetName                   = ''
        self.RTStruct.StructureSetDate                   = datetime.datetime.now().strftime('%Y%m%d')
        self.RTStruct.StructureSetTime                   = datetime.datetime.now().strftime('%H%M%S')

def main(input_RTStructure, input_dicomDataset, output_RTStructure):
    RTStruc       = pydicom.read_file(input_RTStructure)
    DicomDS_files = DFR(input_dicomDataset)
    DicomDS       = []
    for file in DicomDS_files:
        DicomDS.append(pydicom.read_file(file))
    contour_library      = [[] for i in range(len(DicomDS))]
    contour_library[9] = [-230,-150,563.79,    -240,-150,563.79,     -240,-160,563.79,     -230,-160,563.79,     -230,-150,563.79]
    update_ReferencedFrameOfReferenceSequence(RTStruc, DicomDS)
    update_StructureSetROISequence_and_ROIContourSequence(RTStruc, DicomDS, contour_library)
    update_RTStructure_UIDs(RTStruc)
    RTStruc.save_as('Output_RT_Structure.dcm')

import argparse
if __name__ == '__main__' :
  usage = 'Cameron Pain (cameron.pain@austin.org.au): Simple image viewer for TEAP.'
  parser = argparse.ArgumentParser(description = usage)
  parser.add_argument('input_RTStructure' , type = str, help = 'A MIM RT structure to be wiped clean and used as a template.')
  parser.add_argument('input_dicomDataset', type = str, help = 'Path to the directory containing the Dicom data you wish to assign the RT structure to. The directory must contain the single file you want to assign the RTstructure to.')
  parser.add_argument('output_RTStructure', type = str, help = 'The name of the cleaned MIM RT structure you want to save.')
  args = parser.parse_args()
  main(args.input_RTStructure, args.input_dicomDataset, args.output_RTStructure)