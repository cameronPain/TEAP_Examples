#!/usr/bin/env python3
# 20200809
# Calculate NEMA NU2 PET quantitative accuracy: Dead time and randoms.
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

class Accuracy:
    def __init__(self, root_dir, peak_NEC_aconc = 27.461):
        if root_dir[-1]    != '/':
            self.root_dir  = root_dir + '/'
        else:
            self.root_dir  = root_dir
        header_dir         = root_dir + 'headers/'
        data_dir           = root_dir + 'data/'
        header_files       = os.popen('ls ' + header_dir).read().split('\n')[:-1]
        self.header_files  = header_files
        self.header_dir    = header_dir
        self.data_dir      = data_dir
        self.data_parsed   = False
        self.peakNEC_aconc = peak_NEC_aconc

    
    def find_tags(self, dataset, tags):
        values   = []
        for tag in tags:
            val  = None
            nof_char = len(tag)
            for datum in dataset:
                if datum[:nof_char] == tag:
                    val = datum.split(':=')[-1]
                    values.append(val)
                    break
            if val == None:
                print(tag + ' not found')
                values.append(val)
        return values
    
    
    def get_recon_volume(self):
        reference_date            = datetime.datetime.strptime('19000101','%Y%m%d')
        recon_volumes             = []
        acquisition_times         = []
        pixel_spacings            = []
        calibration_times         = []
        half_lifes                = []
        activities                = []
        acquisition_durations     = []
        calibration_factors       = []
        for header in self.header_files: #Iterate through the list of header file names I've pulled from the root_dir
            print(self.header_dir + header)
            print('     Opening byte data')
            volume_header         = open(self.header_dir + header, 'r').read().split('\n')[:-1]
            z_pixel_spacing, y_pixel_spacing, x_pixel_spacing  = self.find_tags(volume_header, ['scale factor (mm/pixel) [3]', 'scale factor (mm/pixel) [2]', 'scale factor (mm/pixel) [1]'])
            z_pixel_spacing, y_pixel_spacing, x_pixel_spacing = float(z_pixel_spacing), float(y_pixel_spacing), float(x_pixel_spacing)

            cal_date,             = self.find_tags(volume_header, ['%tracer injection date (yyyy:mm:dd)'])
            cal_time,             = self.find_tags(volume_header, ['%tracer injection time (hh:mm:ss GMT+00:00)'])
            calibration_time      = datetime.datetime.strptime(cal_date+cal_time,'%Y:%m:%d%H:%M:%S')
            half_life,            = self.find_tags(volume_header, ['isotope gamma halflife (sec)'])
            activity,             = self.find_tags(volume_header, ['tracer activity at time of injection (Bq)'])
            duration,             = self.find_tags(volume_header, ['!image duration (sec)'])
            acq_time,             = self.find_tags(volume_header, ['%study time (hh:mm:ss GMT+00:00)'])
            acq_date,             = self.find_tags(volume_header, ['%study date (yyyy:mm:dd)'])
            acquisition_time      = datetime.datetime.strptime(acq_date + acq_time,'%Y:%m:%d%H:%M:%S')
            x_dim, y_dim, z_dim   = self.find_tags(volume_header, ['matrix size[1]', 'matrix size[2]', 'matrix size[3]'])
            x_dim, y_dim, z_dim   = int(x_dim), int(y_dim), int(z_dim)
            camera_cal_factor,    = self.find_tags(volume_header, ['%scanner quantification factor (Bq*s/ECAT counts)'])
            bytes_per_pixel,      = self.find_tags(volume_header, ['!number of bytes per pixel'])
            if bytes_per_pixel == '4':
                data_type = n.float32
            elif bytes_per_pixel == '2':
                data_type = n.int16
            else:
                data_type = n.int16
                print('Note, the bytes per pixel was not found in the header. It has by default been set as 2.')
            data_file_name,          = self.find_tags(volume_header, ['!name of data file'])
            print(data_file_name)
            print('     Formatting prompt coincidence data into numpy arrays')
            pixel_array              = n.frombuffer(open(self.data_dir + data_file_name, 'rb').read(), dtype = data_type)
            TOF_bin                  = x_dim*y_dim*z_dim
            recon                    = pixel_array.reshape(z_dim,y_dim,x_dim)
            
            """ Decay correct the activity contained in the header accordingly
            """

            decay_time               = (acquisition_time - calibration_time).total_seconds() + float(duration)/2. #Decay correct to the mid point of the acquisition.
            decay_corrected_activity = float(activity) * (2**(-decay_time/float(half_life)))
            recon_volumes.append(recon)
            acquisition_times.append(acquisition_time)
            pixel_spacings.append([z_pixel_spacing, y_pixel_spacing, x_pixel_spacing])
            calibration_times.append(calibration_time)
            half_lifes.append(float(half_life))
            activities.append(float(decay_corrected_activity))
            acquisition_durations.append(float(duration))
            calibration_factors.append(float(camera_cal_factor))

        """ Generate the index set for temporal ordering.
        """
        time_differences           = [ (i - reference_date).total_seconds() for i in acquisition_times]
        index_set                  = []
        for i in range(len(recon_volumes)):
            index_set.append(n.argmin(time_differences))
            time_differences[n.argmin(time_differences)] += time_differences[n.argmax(time_differences)]
    
    
        #Attached the collected data to the object in temporal order.
        self.system_cal_factor     = self.temporal_ordering(calibration_factors, index_set)
        self.recon_volumes         = self.temporal_ordering(recon_volumes, index_set)
        self.acquisition_times     = self.temporal_ordering(acquisition_times, index_set)
        self.pixel_spacings        = self.temporal_ordering(pixel_spacings, index_set)
        self.calibration_times     = self.temporal_ordering(calibration_times, index_set)
        self.half_lifes            = self.temporal_ordering(half_lifes, index_set)
        self.activities            = self.temporal_ordering(activities, index_set)
        self.acquisition_durations = self.temporal_ordering(acquisition_durations, index_set)
        


    #The data are originally loaded in by alpha-numeric ordering which isnt necessarily the temporal ordering. This orders the data based on some index set which I chose as temporal ordering.
    def temporal_ordering(self, set, index_set):
        ordered_set = []
        for i in index_set:
            ordered_set.append(set[i])
        return ordered_set
    



    """ This method generates a .pdf file with the reported values as specified in NEMA NU2 2007.
    """
    def generate_report(self, save_directory = '', file_name = 'NEMA_NU2_Accuracy', system_details = ['Siemens', 'Intevo'], captions = [''], delete_tex_files = True):
        #Generate Plots
        fontsize = 26
        fig, (ax0,ax1,ax2) = pyplot.subplots(3,1, figsize=(21,21))
        ax0.scatter(self.activity_conc , self.acquisition_rate, color='b',   label = 'Measured Count Rate'  )
        ax0.plot(self.activity_conc    , self.acquisition_rate, color='b')
        ax0.plot(self.activity_conc    , self.extrap_rate     , '--', color='b', label = 'Extrapolated Count Rate')
        ax0.legend(fontsize=fontsize)
        ax0.grid('on')
        ax0.tick_params(axis='both', labelsize=fontsize)
        ax0.yaxis.offsetText.set_fontsize(fontsize)
        ax0.set_ylabel('Event rate ($s^{-1}$)', fontsize=fontsize)
        #ax0.set_ylabel('Event Rate ($ms^{-1}$)', fontsize=16)
        #ax0.set_xlabel('Activity concentration ($kBq/ml$)', fontsize=16)
        ax1.scatter(self.activity_conc, self.max_deadtime, label = 'Maximum dead time percentage', color='r')
        ax1.scatter(self.activity_conc, self.min_deadtime, label = 'Minimum dead time percentage', color='b')
        ax1.plot(self.activity_conc   , self.max_deadtime, color='r')
        ax1.plot(self.activity_conc   , self.min_deadtime, color='b')
        ax1.plot([self.peakNEC_aconc, self.peakNEC_aconc], [n.amin(self.min_deadtime), n.amax(self.max_deadtime)],'--', color='k', lw=2.0, label = 'Maximum NEC')
        ax1.plot([0,n.amax(self.activity_conc)], [0,0], color='k', lw=1.5)
        ax1.legend(fontsize=fontsize)
        ax1.tick_params(axis='both', labelsize=fontsize)
        ax1.set_ylim((n.amin(self.min_deadtime)*0.99, n.amax(self.max_deadtime)*1.01))
        ax1.yaxis.offsetText.set_fontsize(fontsize)
        ax1.grid('on')
        ax1.set_xlabel('Activity Concentration ($kBq/ml$)', fontsize=fontsize)
        
        ax2.scatter(n.arange(len(self.deadtime_slice_peakNEC)), abs(self.deadtime_slice_peakNEC), color='b', label = 'Dead time percentage at NEC')
        ax2.plot(n.arange(len(self.deadtime_slice_peakNEC)), abs(self.deadtime_slice_peakNEC), color='b')
        ax2.set_ylabel('Dead time percentage', fontsize=fontsize )
        ax2.set_xlabel('Slice Index', fontsize=fontsize)
        ax2.tick_params(axis='both', labelsize=fontsize)
        ax2.legend(fontsize=fontsize)

        ax1.set_ylabel('Dead time percentage', fontsize=fontsize)
        pyplot.tight_layout()
        #ax1.set_ylabel('Event Rate ($ms^{-1}$)', fontsize=16)
        #ax1.set_xlabel('Activity concentration ($kBq/ml$)', fontsize=16)

        #fig.text(0.42, 0.05, 'Activity concentration ($kBq/ml$)', fontsize=fontsize)
        self.count_rate_plot = fig
        geometry_options = {"tmargin": "1cm", "lmargin": "2cm", "rmargin": "2cm"}
        pylatex_document = Document(geometry_options = geometry_options)
        #Build the header to the document.
        with pylatex_document.create(Tabu("X[l] ")) as first_page_table:
            my_details = MiniPage(width=NoEscape(r"0.6\textwidth"), pos='t!', align='l')
            my_details.append(bold('Auto-generated NEMA NU2 Dead time losses'))
            my_details.append(LineBreak())
            my_details.append(bold(NoEscape(r"Generated on \today")))
            my_details.append(NoEscape(r"\vspace{0.3cm}"))
            my_details.append(LineBreak())
            my_details.append('Cameron Pain')
            my_details.append(LineBreak())
            my_details.append('cameron.pain@austin.org.au')
        first_page_table.add_row([my_details])
        first_page_table.add_empty_row()
        #Put in the system details. This is limited at this point.
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
        pylatex_document.append(NoEscape(r"\vspace{1cm}"))
        #Put in a table with the reported count rate values specified in the NEMA NU2 2007 document.
        """
        with pylatex_document.create(Tabu("X[l] X[c] X[c]")) as table:
            #table.add_caption("Peak true coincidence rate and peak noise equivalent count rate and the corresponding activity concentrations for which they are measured.")
            row1 = ["",NoEscape(r"maximum value ($ms^{-1}$)"),NoEscape(r"activity concentration ($kBq/ml$)")]
            row2 = ["True", str(n.round(n.amax(self.True_Rate), 3)), str( n.round(self.activity_conc[n.argmax(self.True_Rate)], 3))]
            row3 = ["NECR", str(n.round(n.amax(self.NEC_Rate), 3)), str( n.round(self.activity_conc[n.argmax(self.NEC_Rate)], 3))]
            row4 = ["NECR maximum scatter fraction", str(n.round(self.Scatter_Fraction[n.argmax(self.NEC_Rate)], 3)), str( n.round(self.activity_conc[n.argmax(self.NEC_Rate)], 3))]
            table.append(NoEscape(r"\hline"))
            table.add_row(row1)
            table.append(NoEscape(r"\hline"))
            table.add_row(row2)
            table.add_row(row3)
            table.add_row(row4)
            table.append(NoEscape(r"\hline"))
        """
        pyplot.figure(self.count_rate_plot.number)
        #Put in the count rate figure.
        with pylatex_document.create(Figure(position = '!h')) as figure:
            figure.add_plot(width = NoEscape(r"\textwidth"))
            figure.add_caption(captions[0])
        #Put in the scatter fraction figure.
        pylatex_document.generate_pdf(file_name, clean_tex=delete_tex_files)
        self.LaTex_File = pylatex_document



    def __get_inner_65cm(self, image, z_pixel_spacing):
        z_fov    = z_pixel_spacing*len(image)
        trim_len = z_fov - 650.0 # mm
        if trim_len <= 0:
            #print('The axial FOV is < 65 cm. We do not need to trim it.')
            return n.array(image)
        else:
            trim_slices = int(trim_len/(2*z_pixel_spacing))
            return n.array(image[trim_slices:-trim_slices])


    def __get_circle_ROI_mask(self, radius = 90): #90 mm
        print('Generating cylindrical mask...')
        recon_volume_dims              = n.shape(self.recon_volumes)
        ROI_template                   = []
        #Pick out the params for the first acquisition and assume they apply to all acquisitions. Less general, but it prevents us having to build a new template each time.
        pixel_dims                     = self.pixel_spacings[0]
        zero_slice                     = n.zeros(recon_volume_dims[-2:])
        centre_coord_x, centre_coord_y = recon_volume_dims[-2]//2, recon_volume_dims[-1]//2
        centre_mm_x, centre_mm_y       = centre_coord_x*pixel_dims[2], centre_coord_y*pixel_dims[1]
        print('Generating binary mask for volume...')
        print('     creating 2D mask...')
        for i in range(recon_volume_dims[-2]): #Create binary mask with all pixels which are outside the 180 mm diameter ROI centred on the centre pixel set to zero and one otherwise.
            for j in range(recon_volume_dims[-1]):
                i_mm, j_mm = i*pixel_dims[1], j*pixel_dims[2]
                pos        = n.sqrt( (centre_mm_x - j_mm)**2 + (centre_mm_y - i_mm)**2)
                if pos <= radius:
                    zero_slice[i,j] = 1
                else:
                    pass
        print('     stacking 2D mask for volume mask...')
        #Stack up the zero slice templates to create a volume mask.
        for i in range(recon_volume_dims[0]): #for each acquisition
            acquisition_template = []
            for j in range(recon_volume_dims[1]): #for each slice in the ith acquisition
                    acquisition_template.append(zero_slice)
            ROI_template.append(acquisition_template)
        ROI_template = n.array(ROI_template)
        return ROI_template
    
    def analyse_data(self):
        Phantom_Volume          = 22000 # ml
        self.get_recon_volume()
        self.activity_conc          = n.divide(self.activities, 1000*Phantom_Volume) #kBq/ml
        mask                    = self.__get_circle_ROI_mask()
        masked_volumes          = n.multiply(mask, self.recon_volumes)
        print(n.shape(masked_volumes))
        true_rate               = n.sum(n.sum(masked_volumes,axis=-1), axis=-1)
        intr_rate               = true_rate[-3:]
        intr_activities         = self.activities[-3:]
        efficiency              = n.divide(intr_rate.transpose(), intr_activities)
        ave_efficiency          = n.mean(efficiency, axis=1)
        deadtime_percent        = []
        for i in range(len(masked_volumes)):
            #Call a bunch of the "protected" methods to correctly process the data for analysis.
            rate_intrin     = n.multiply(self.activities[i], ave_efficiency)
            deadtime_percent.append( 100*( n.subtract( n.divide(true_rate[i], rate_intrin), 1)))
        deatime_at_NEC          = []
        peakNEC_index           = n.argmin( abs( n.subtract(self.activity_conc, self.peakNEC_aconc) )  )

        """
        for i in range(len(masked_volumes)):
            for j in range(len(masked_volumes[i])):
                pyplot.imshow(masked_volumes[i][j],vmin = 0.0, vmax = n.amax(masked_volumes[i]))
                pyplot.show()
        """
        min_vals  = []
        max_vals  = []
        start     = 2
        stop      = -2
        for line in deadtime_percent:
            min_vals.append(n.amin(line[start:stop]))
            max_vals.append(n.amax(line[start:stop]))
        
        self.deadtime_slice_peakNEC = deadtime_percent[peakNEC_index]
        self.min_deadtime           = min_vals
        self.max_deadtime           = max_vals
        self.acquisition_rate       = n.mean(true_rate, axis=-1)
        self.extrap_rate            = n.multiply(self.activities, n.mean(ave_efficiency, axis=-1))
        self.deadtime_percent       = deadtime_percent


def main(root_dir):
    #If ran as main, it will analyse a specified root_dir.
    accuracy = Accuracy(root_dir)
    accuracy.analyse_data()
    accuracy.generate_report(captions = ['Quantitative measured of dead time performance. (1) The measured total count rate and the linear extrapolation from low count rate. (2) Maximum and minium percentage variations over all slices between measured and extrapolated count rate. (3) Dead time percentage difference for each slice at peak NEC.'], delete_tex_files = True)

import argparse
if __name__  == '__main__':
    usage  = 'cameron.pain@austin.org.au 20200808: Calculate the NEMA NU2 sensitivity.'
    parser = argparse.ArgumentParser(usage = usage)
    parser.add_argument('root_dir', type = str, help = 'Directory containing two directories named "headers" and "data" containing the singogram headers and data files respectively.')
    args = parser.parse_args()
    main(args.root_dir)




