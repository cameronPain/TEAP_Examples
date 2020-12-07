#!/usr/bin/env python3
# 20200918
# Calculate NEMA NU2 PET camera characteristics
#
#
#
import numpy as n
import matplotlib.pyplot as pyplot
from matplotlib.widgets import Slider
import pydicom as pydicom
import os
import datetime
from scipy.optimize import curve_fit as curve_fit
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, Plot, Figure, Matrix, Alignat, PageStyle, Head, Tabu, MiniPage, Foot, LineBreak
from pylatex.utils import italic, NoEscape, bold
import sys

#Import the relevant local classes.
workingDir = os.popen('pwd').read().split('\n')[0]
sys.path.append(workingDir + '/Imports/')
print(workingDir + '/Imports/' + ' appended to sys.path')
from NEMA_NU2_SpatialResolution import SpatialResolution
from NEMA_NU2_Scatter           import Scatter
from NEMA_NU2_Sensitivity       import Sensitivity
from NEMA_NU2_Accuracy          import Accuracy

class NEMA_NU2_2007:
    def __init__(self, root_dir):
        if root_dir[-1]       != '/':
            self.root_dir      = root_dir + '/'
        else:
            self.root_dir      = root_dir
        #Get all the subdirs for each test
        self.scatter_dir            = root_dir + 'scatter/'
        self.sensitivity_dir_0cm    = root_dir + 'sensitivity/0cm/'
        self.sensitivity_dir_10cm   = root_dir + 'sensitivity/10cm/'
        self.spatial_resolution_dir = root_dir + 'spatial_resolution/'
        self.image_quality_dir      = root_dir + 'image_quality/'
        self.accuracy_dir           = root_dir + 'accuracy/'

    def analyse_data(self):
        print('     |-------------------------------------------------------------')
        print('     |NEMA NU2 2007: Spatial Resolution')
        print('     |-------------------------------------------------------------')
        #Calculate the scatter properties.
        self.SpatialResolution                           = SpatialResolution(self.spatial_resolution_dir)
        self.SpatialResolution.analyse_data()
        self.SpatialResolution.generate_report(file_name = 'Results/NEMA_NU2_SpatialResolution', system_details = ['Siemens','Epworth','Tested: 28/05/2020'])
        
        
        print('     |-------------------------------------------------------------')
        print('     |NEMA NU2 2007: Scatter Fraction, Count Losses And Randoms Measurement')
        print('     |-------------------------------------------------------------')
        #Calculate the scatter properties.
        self.Scatter                           = Scatter(self.scatter_dir)
        self.Scatter.analyse_data()
        self.Scatter.generate_report(file_name = 'Results/NEMA_NU2_Scatter', system_details = ['Siemens','Epworth','Tested: 28/05/2020'], captions = ['Count rate performance and scatter fraction with system randoms estimate.','Count rate performance and scatter fraction with low count rate approximation randoms estimate.'])
        
        print('     |-------------------------------------------------------------')
        print('     |NEMA NU2 2007: Sensitivity 0cm')
        print('     |-------------------------------------------------------------')
        #Calculate the senstivity.
        self.Sensitivity                           = Sensitivity(self.sensitivity_dir_0cm)
        self.Sensitivity.get_image_data()
        self.Sensitivity.calculate_sensitivity(show_images = False)
        self.Sensitivity.generate_report(file_name = 'Results/NEMA_NU2_Sensitivity_0cm', system_details = ['Siemens', '0cm radial offset','Epworth','Tested: 28/05/2020'], captions = ['Systems sensitivity as a function of attenuator thickness and the axial sensitivity profile. Reported data has been random subtracted.',''])
        print('     |-------------------------------------------------------------')
        print('     |NEMA NU2 2007: Sensitivity 10cm')
        print('     |-------------------------------------------------------------')
        #Calculate the senstivity.
        self.Sensitivity                           = Sensitivity(self.sensitivity_dir_10cm)
        self.Sensitivity.get_image_data()
        self.Sensitivity.calculate_sensitivity(show_images = False)
        self.Sensitivity.generate_report(file_name = 'Results/NEMA_NU2_Sensitivity_10cm', system_details = ['Siemens', '10cm radial offset','Epworth','Tested: 28/05/2020'], captions = ['Systems sensitivity as a function of attenuator thickness and the axial sensitivity profile. Reported data has been random subtracted.',''])

            
        print('     |-------------------------------------------------------------')
        print('     |NEMA NU2 2007: Accuracy Corrections for Count Losses and Randoms')
        print('     |-------------------------------------------------------------')
        #Calculate the scatter properties.
        self.accuracy                           = Accuracy(self.accuracy_dir, peak_NEC_aconc = self.Scatter.peak_NEC_aconc)
        self.accuracy.analyse_data()
        self.accuracy.generate_report(file_name = 'Results/NEMA_NU2_Accuracy', system_details = ['Siemens','Epworth','Tested: 28/05/2020'], captions = ['Quantitative measured of dead time performance. (1) The measured total count rate and the linear extrapolation from low count rate. (2) Maximum and minium percentage variations over all slices between measured and extrapolated count rate. (3) Dead time percentage difference for each slice at peak NEC.'], delete_tex_files = True)





def main(root_dir):
    print('|-------------------------------------------------------------')
    print('|NEMA NU2 2007: Performance Measurements of Positron Emission Tomographs')
    print('|Cameron Pain 20200918: cameron.pain@austin.org.au')
    print('|-------------------------------------------------------------')
    NEMA_Analysis = NEMA_NU2_2007(root_dir)
    NEMA_Analysis.analyse_data()


import argparse
if __name__  == '__main__':
    usage  = 'cameron.pain@austin.org.au 20200808: Calculate the NEMA NU2 sensitivity.'
    parser = argparse.ArgumentParser(usage = usage)
    parser.add_argument('root_dir', type = str, help = 'Directory containing two directories named "headers" and "data" containing the singogram headers and data files respectively.')
    args = parser.parse_args()
    main(args.root_dir)




