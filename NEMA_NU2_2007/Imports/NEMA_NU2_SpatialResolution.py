#!/usr/bin/env python3
# 20200809
# Calculate NEMA NU2 PET sensitivity for a Siemens system. Defines a sensitivity class which you give the root directory path which contains two subdirectories: headers/ and data/
# data/ contains the byte data files and headers/ contains the header files for the byte data. The code formats the byte data into images using the information in the header file (Assumes a 16 bit integer at this point) and sums the data accordingly. Generates the decay corrected count rate data and fits the NEMA NU2 specified exponential fit as a function of shielding thickness.
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
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, Plot, Figure, Matrix, Alignat, PageStyle, Head, Tabu, MiniPage, Foot, LineBreak, NewPage
from pylatex.utils import italic, NoEscape, bold



class SpatialResolution:
    def __init__(self, root_dir):
        #Stick a / on the end so it can be used in a file path.
        if root_dir[-1]!= '/':
            root_dir += '/'
        else:
            pass
        #Header directory
        self.header_dir  = root_dir + 'headers/'
        #Data directory
        self.data_dir    = root_dir + 'data/'
        #Create a list of source position names which we can iterate through to get the measurements at different locations.
        positions        = ['0_1_0','0_10_0','0_20_0','20_1_0','20_10_0','20_20_0'] #The (x,y,z) position of the source is specified as x_y_z. The 1/8 FoV is written as 20 just as a default.
        #Save the positions so we can assign the position to each file more easily.
        self.positions   = positions
        #Create containers for each of the files names so we can iterate through and open the data files as we need them.
        header_files     = []
        data_files       = []
        #For each position.
        for position in positions:
            #get the header file name in the position directory
            file        = os.popen('ls ' + self.header_dir + position + '/').read().split('\n')[0]
            #append the file path to the header file.
            header_files.append(self.header_dir + position + '/' + file)
            #Temporarily open the header so we can pull out it's corresponding data file path.
            temp_header = open(self.header_dir + position + '/' + file, 'r').read().split('\n')[:-1]
            data_file,  = self.find_tags(temp_header, ['!name of data file'])
            #Stick the data file path into the data_files container
            data_files.append(self.data_dir + position + '/' + data_file)
            #Clean up to save memory.
            del(temp_header)
            del(data_file)
            del(file)
        #Save the file paths to the object for later.
        self.header_files = header_files
        self.data_files   = data_files

    def gaussian_fit(self, x, x0, A, sigma):
        return A*n.exp( -((x-x0)**2)/(2*(sigma**2)))
    
    def analyse_data(self):
        #Generating line projections to get max coordinates.
        #Create containers to save figures generated and float values for the FWHM and FWTM's we get.
        images      = []
        resolutions = []
        print('Calculating spatial resolution:')
        for i in range(len(self.header_files)):
            print(self.header_files[i])
            #Open the header and split by a new line.
            header     = open(self.header_files[i], 'r').read().split('\n')
            #Use the find_tags method I made to pull out the corresponding values to a list of tag names.
            dims       = self.find_tags(header, ['matrix size[3]','matrix size[2]','matrix size[1]'], data_type = int)
            #Open up the corresponding volume file.
            data       = open(self.data_files[i], 'rb').read()
            z_pix, y_pix, x_pix = self.find_tags(header, ['scale factor (mm/pixel) [3]', 'scale factor (mm/pixel) [2]','scale factor (mm/pixel) [1]'])
            z_pix, y_pix, x_pix = float(z_pix), float(y_pix), float(x_pix)
            #Pull in image data from the bytes. Should change the data type based on the bytes per value contained in the header. It's a float value but this isn't quite as robust.
            image_data = n.frombuffer(data, n.float32).reshape(dims)
            self.image = image_data
            self.z_pix = z_pix
            self.x_pix = x_pix
            self.y_pix = y_pix
            #Create projections in two dimensions so that I can find maxima along a line profile to locate the point.
            z_proj = n.sum(self.image, axis=0)
            y_proj = n.sum(self.image, axis=1)
            x_proj = n.sum(self.image, axis=2)
            xz_proj = n.sum(z_proj,axis=0)
            yz_proj = n.sum(z_proj,axis=1)
            xy_proj = n.sum(y_proj,axis=1)
            #Getting max coords
            #Just use a simple argmax to get the maxima. There shouldn't be any problems with this based on the simplicity of the source distribution.
            z_max, y_max, x_max = n.argmax(xy_proj), n.argmax(yz_proj), n.argmax(xz_proj)
            self.max_coords = n.array([z_max, y_max, x_max])
            #Cropping the point image out with a manually specified im_range.
            im_range = 10
            im_crop  = self.image[z_max-im_range:z_max+im_range, y_max-im_range:y_max+im_range, x_max-im_range:x_max+im_range]
            im_max   = n.amax(im_crop)
            self.crop_image = im_crop
            #Get profiles through the maximum pixel values in the crop images.
            self.x_profile = self.crop_image[im_range,im_range]
            self.y_profile = self.crop_image[im_range,:,im_range]
            self.z_profile = self.crop_image[:,im_range,im_range]
            #Get Gaussian parameters fit to the profile data.
            z_params,cov = curve_fit(self.gaussian_fit, n.arange(len(self.z_profile)), self.z_profile, p0=[len(self.z_profile)/2.0, n.amax(self.z_profile), 3.0])
            y_params,cov = curve_fit(self.gaussian_fit, n.arange(len(self.y_profile)), self.y_profile, p0=[len(self.y_profile)/2.0, n.amax(self.y_profile), 3.0])
            x_params,cov = curve_fit(self.gaussian_fit, n.arange(len(self.x_profile)), self.x_profile, p0=[len(self.x_profile)/2.0, n.amax(self.x_profile), 3.0])
            z_fit_range = n.multiply(n.arange(0, len(self.z_profile),0.1),self.z_pix)
            y_fit_range = n.multiply(n.arange(0, len(self.y_profile),0.1),self.y_pix)
            x_fit_range = n.multiply(n.arange(0, len(self.x_profile),0.1),self.x_pix)
            self.z_params = z_params
            self.y_params = y_params
            self.x_params = x_params
            #Generating plots.
            coordinate_offset = n.array([len(self.image)//2, len(self.image[0])//2, len(self.image[0][0])//2])
            #print('FOV Size ', n.multiply(coordinate_offset, n.multiply(2, [self.z_pix, self.y_pix, self.x_pix] )))
            f, (ax0, ax1, ax2) = pyplot.subplots(1,3,figsize=(18,6))
            f.text(0.01,0.95, '[x,y,z]: ' + str(self.positions[i].split('_')) , fontsize=16)
            ax0.plot(x_fit_range*self.x_pix, self.gaussian_fit(x_fit_range, *x_params),'--', color='k')
            ax0.scatter(n.arange(len(self.crop_image))*self.x_pix, self.crop_image[im_range,im_range])
            ax0.set_xlabel('Profile through point ($mm$)', fontsize=16)
            ax0.set_ylabel('Counts', fontsize=14)
            ax0.set_title('x profile\nFWHM: ' + str( n.round(  x_params[-1]*2*n.sqrt(2*n.log(2))*self.x_pix,3)) + ' mm\n FWTM: ' + str( n.round(  x_params[-1]*2*n.sqrt(2*n.log(10))*self.x_pix ,3)) + ' mm', fontsize=14)
            ax0.set_xlim((n.amin(n.arange(len(self.crop_image))*self.x_pix), n.amax(n.arange(len(self.crop_image))*self.x_pix)))
            ax0.tick_params(axis='both',labelsize=14)
            ax1.plot(y_fit_range*self.y_pix, self.gaussian_fit(y_fit_range, *y_params),'--', color='k')
            ax1.scatter(n.arange(len(self.crop_image))*self.y_pix, self.crop_image[im_range,:,im_range])
            ax1.set_xlabel('Profile through point ($mm$)', fontsize=16)
            #ax1.set_ylabel('Counts', fontsize=14)
            ax1.set_title('y profile\nFWHM: ' + str( n.round(  y_params[-1]*2*n.sqrt(2*n.log(2))*self.y_pix,3)) + ' mm\n FWTM: ' + str( n.round(   y_params[-1]*2*n.sqrt(2*n.log(10))*self.y_pix,3)) + ' mm', fontsize=14)
            ax1.set_xlim((n.amin(n.arange(len(self.crop_image))*self.y_pix), n.amax(n.arange(len(self.crop_image))*self.y_pix)))
            ax1.tick_params(axis='both',labelsize=14)
            ax2.plot(z_fit_range*self.z_pix, self.gaussian_fit(z_fit_range, *z_params),'--', color='k')
            ax2.scatter(n.arange(len(self.crop_image))*self.z_pix, self.crop_image[:,im_range,im_range])
            ax2.set_xlabel('Profile through maximum ($mm$)', fontsize=14)
            #ax2.set_ylabel('Counts')
            ax2.set_title('z profile\nFWHM: ' + str( n.round( z_params[-1]*2*n.sqrt(2*n.log(2))*self.z_pix,3)) + ' mm\n FWTM: ' + str( n.round( z_params[-1]*2*n.sqrt(2*n.log(10))*self.z_pix,3)) + ' mm', fontsize=14)
            ax2.set_xlim((n.amin(n.arange(len(self.crop_image))*self.z_pix), n.amax(n.arange(len(self.crop_image))*self.z_pix)))
            ax2.tick_params(axis='both',labelsize=14)
            images.append(f)
            resolutions.append([[n.round(x_params[-1]*2*n.sqrt(2*n.log(2))*self.x_pix,3),n.round(x_params[-1]*2*n.sqrt(2*n.log(10))*self.x_pix,3)],[n.round(  y_params[-1]*2*n.sqrt(2*n.log(2))*self.y_pix,3),n.round(  y_params[-1]*2*n.sqrt(2*n.log(10))*self.y_pix,3)],[n.round( z_params[-1]*2*n.sqrt(2*n.log(2))*self.z_pix,3),n.round( z_params[-1]*2*n.sqrt(2*n.log(10))*self.z_pix,3)]])
        self.figures     = images
        self.resolutions = resolutions
        
        
        
    def find_tags(self, dataset, tags, data_type = str):
        values   = []
        for tag in tags:
            val  = None
            nof_char = len(tag)
            for datum in dataset:
                if datum[:nof_char] == tag:
                    val = datum.split(':=')[-1]
                    values.append(data_type(val))
                    break
            if val == None:
                print(tag + ' not found')
                values.append(data_type(val))
        return values
                
    def generate_report(self, save_directory = '', file_name = 'NEMA_NU2_Spatial_Resolution', system_details = ['Siemens', 'Intevo'], captions = ['','']):
        geometry_options = {"tmargin": "1cm", "lmargin": "1cm", "rmargin": "1cm"}
        pylatex_document = Document(geometry_options = geometry_options)
        with pylatex_document.create(Tabu("X[l] ")) as first_page_table:
               my_details = MiniPage(width=NoEscape(r"0.6\textwidth"), pos='t!', align='l')
               my_details.append(bold('Auto-generated NEMA NU2 Spatial Resolution Report'))
               my_details.append(LineBreak())
               my_details.append(bold(NoEscape(r"Generated on \today")))
               my_details.append(NoEscape(r"\vspace{0.3cm}"))
               my_details.append(LineBreak())
               my_details.append('Cameron Pain')
               my_details.append(LineBreak())
               my_details.append('cameron.pain@austin.org.au')
               first_page_table.add_row([my_details])
               first_page_table.add_empty_row()
        with pylatex_document.create(Tabu("X[r] ")) as first_page_table:
               pylatex_document.append(NoEscape(r"\vspace{-2.5cm}"))
               my_details = MiniPage(width=NoEscape(r"0.2\textwidth"), pos='t!', align='l')
               for i in range(len(system_details)-1):
                  my_details.append(system_details[i])
                  my_details.append(LineBreak())
               my_details.append(system_details[-1])
               first_page_table.add_row([my_details])
               first_page_table.add_empty_row()
        pylatex_document.append(NoEscape(r"\vspace{-1cm}"))
        pylatex_document.append(NoEscape(r"\hrule"))
        #pylatex_document.append(NoEscape(r"\vspace{-1cm}"))
        for i in [0,1]:
            with pylatex_document.create(Figure(position = 'h')) as figure:
                for j in range(i*3, (i+1)*3):
                    pyplot.figure(self.figures[j].number)
                    figure.add_plot(width = NoEscape(r"0.90\textwidth"))
                    figure.append(LineBreak())
        pylatex_document.generate_pdf(file_name, clean_tex=True)
        self.LaTex_File = pylatex_document
        
def main(root_dir):
    spatRes = SpatialResolution(root_dir)
    spatRes.find_maxima()
    spatRes.generate_report()

import argparse
if __name__  == '__main__':
    usage  = 'cameron.pain@austin.org.au 20200808: Calculate the NEMA NU2 spatial resolution.'
    parser = argparse.ArgumentParser(usage = usage)
    parser.add_argument('root_dir', type = str, help = 'Directory containing the resolution data directories.')
    args = parser.parse_args()
    main(args.root_dir)




