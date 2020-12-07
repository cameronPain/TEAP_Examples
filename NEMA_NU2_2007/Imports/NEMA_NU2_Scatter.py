#!/usr/bin/env python3
# 20200809
# Calculate NEMA NU2 PET count losses and scatter data.
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



class Scatter:
    def __init__(self, root_dir):
        if root_dir[-1]   != '/':
            self.root_dir = root_dir + '/'
        else:
            self.root_dir = root_dir
        header_dir        = root_dir + 'headers/'
        data_dir          = root_dir + 'data/'
        header_files      = os.popen('ls ' + header_dir).read().split('\n')[:-1]
        self.header_files = header_files
        self.header_dir   = header_dir
        self.data_dir     = data_dir
        self.data_parsed  = False

    
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
    
    def get_sum_sinogram_image(self):
        reference_date        = datetime.datetime.strptime('19000101','%Y%m%d')
        TOF_Oblique_Binned    = []
        Randoms               = []
        acquisition_times     = []
        pixel_spacings        = []
        calibration_times     = []
        half_lifes            = []
        activities            = []
        acquisition_durations = []
        for header in self.header_files: #Iterate through the list of header file names I've pulled from the root_dir
            print(self.header_dir + header)
            print('     Opening byte data')
            sinogram_header       = open(self.header_dir + header, 'r').read().split('\n')[:-1]
            oblique_binning,      = self.find_tags(sinogram_header , ['%segment table'])
            obq_split             = oblique_binning.split(',')
            obq                   = []
            for i in obq_split:
                a1 = i.replace('{','')
                a2 = a1.replace('}','')
                a3 = int(a2)
                obq.append(a3)
            z_pixel_spacing, y_pixel_spacing, x_pixel_spacing  = self.find_tags(sinogram_header, ['scale factor (mm/pixel) [3]', 'scale factor (degree/pixel) [2]', 'scale factor (mm/pixel) [1]'])
            z_pixel_spacing, y_pixel_spacing, x_pixel_spacing = float(z_pixel_spacing), float(y_pixel_spacing), float(x_pixel_spacing)
            oblique_segments      = obq # I think this is the binning of oblique sinograms.
            cal_date,             = self.find_tags(sinogram_header, ['%tracer injection date (yyyy:mm:dd)'])
            cal_time,             = self.find_tags(sinogram_header, ['%tracer injection time (hh:mm:ss GMT+00:00)'])
            calibration_time      = datetime.datetime.strptime(cal_date+cal_time,'%Y:%m:%d%H:%M:%S')
            half_life,            = self.find_tags(sinogram_header, ['isotope gamma halflife (sec)'])
            activity,             = self.find_tags(sinogram_header, ['tracer activity at time of injection (Bq)'])
            duration,             = self.find_tags(sinogram_header, ['%image duration from timing tags (msec)'])
            acq_time,             = self.find_tags(sinogram_header, ['%study time (hh:mm:ss GMT+00:00)'])
            acq_date,             = self.find_tags(sinogram_header, ['%study date (yyyy:mm:dd)'])
            acquisition_time      = datetime.datetime.strptime(acq_date + acq_time,'%Y:%m:%d%H:%M:%S')
            x_dim, y_dim, z_dim   = self.find_tags(sinogram_header, ['matrix size[1]', 'matrix size[2]', 'matrix size[3]'])
            x_dim, y_dim, z_dim   = int(x_dim), int(y_dim), int(z_dim)
            bytes_per_pixel,      = self.find_tags(sinogram_header, ['!number of bytes per pixel'])
            data_file_name,       = self.find_tags(sinogram_header, ['!name of data file'])
            nof_tof_bins,         = self.find_tags(sinogram_header, ['%number of TOF time bins'])
            print('     Formatting prompt coincidence data into numpy arrays')
            pixel_array           = n.frombuffer(open(self.data_dir + data_file_name, 'rb').read(), dtype = n.int16)
            TOF_bin               = x_dim*y_dim*z_dim
            prompt_images         = []
            for i in range(int(nof_tof_bins)):
                prompt_images.append(pixel_array[TOF_bin*i:TOF_bin*(i+1)].reshape(z_dim,y_dim,x_dim))
            print('     Formatting randoms data into numpy array')
            randoms_image = pixel_array[TOF_bin*int(nof_tof_bins):TOF_bin*(int(nof_tof_bins)+1)].reshape([z_dim,y_dim,x_dim])
            randoms_image = n.array(randoms_image)
            #---------------------------------------------------------------------------------------------------------------#
            """ This code here stacks the 3d image matrix according to the segment table in the header. The segment table specifies the bounds between oblique sinograms which are stuck all concatenated into the base image. I should probably stick the process into a method and call the method twice instead of hard coding like this."""

            #prompt.append(prompt_images)
            count_images      = []
            TOF_sum = n.sum(prompt_images,axis=0)
            #Put each oblique sinogram into a separate container
            for i in range(len(oblique_segments)):
                ct = []
                start_bound = int(n.sum(oblique_segments[:int(i)]))
                end_bound   = start_bound + int(n.sum(oblique_segments[int(i)]))
                for j in range(start_bound, end_bound):
                    ct.append(TOF_sum[j])
                count_images.append(ct)
            #pull the first sinogram bin out as the base image.
            base_image = count_images[0]
            #add the oblique sinograms into the base image around the centre pixel.
            for i in range(1,len(count_images)-1):
                mid_index   = len(base_image)//2
                start_index = int(mid_index - (oblique_segments[i]//2))
                end_index   = int(mid_index + (oblique_segments[i]//2))
                for j in range(start_index, end_index):
                    base_image[j] += count_images[i][j - start_index]
            TOF_Oblique_Binned.append(base_image)
            
            #Repeat this process for the randoms image.
            random_images = []
            for i in range(len(oblique_segments)):
                ct = []
                start_bound = int(n.sum(oblique_segments[:int(i)]))
                end_bound   = start_bound + int(n.sum(oblique_segments[int(i)]))
                for j in range(start_bound, end_bound):
                    ct.append(randoms_image[j])
                random_images.append(ct)
            #pull the first sinogram bin out as the base image.
            random_base = random_images[0]
            #add the oblique sinograms into the base image around the centre pixel.
            for i in range(1,len(random_images)-1):
                mid_index   = len(random_base)//2
                start_index = int(mid_index - (oblique_segments[i]//2))
                end_index   = int(mid_index + (oblique_segments[i]//2))
                for j in range(start_index, end_index):
                    random_base[j] += random_images[i][j - start_index]
            Randoms.append(n.array(random_base))
            #---------------------------------------------------------------------------------------------------------------#

            """ Decay correct the activity contained in the header accordingly
            """
            decay_time   = (acquisition_time - calibration_time).total_seconds() + float(duration)/2000. #Decay correct to the mid point of the acquisition.
            decay_corrected_activity = float(activity) * (2**(-decay_time/float(half_life)))

            acquisition_times.append(acquisition_time)
            pixel_spacings.append([z_pixel_spacing, y_pixel_spacing, x_pixel_spacing])
            calibration_times.append(calibration_time)
            half_lifes.append(float(half_life))
            activities.append(float(decay_corrected_activity))
            acquisition_durations.append(float(duration))

        """ Generate the index set for temporal ordering.
        """
        time_differences           = [ (i - reference_date).total_seconds() for i in acquisition_times]
        index_set                  = []
        for i in range(len(TOF_Oblique_Binned)):
            index_set.append(n.argmin(time_differences))
            time_differences[n.argmin(time_differences)] += time_differences[n.argmax(time_differences)]
        #Attached the collected data to the object in temporal order.
        self.TOF_Oblique_Binned    = self.temporal_ordering(TOF_Oblique_Binned, index_set)
        self.Randoms               = self.temporal_ordering(Randoms, index_set)
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
    
    #This method is used if you want to pull in TOF data which hasnt been summed together. It isn't used really because it stores 13x the memory as it loads in all the TOF bins into separate 3d images.
    def get_all_image_data(self):
        self.data_parsed   = True
        prompt             = []
        random             = []
        acquisition_times  = []
        TOF_Oblique_Binned = []
        for header in self.header_files:
            print('Opening byte data')
            sinogram_header       = open(self.header_dir + header, 'r').read().split('\n')[:-1]
            oblique_binning,      = self.find_tags(sinogram_header , ['%segment table'])
            obq_split             = oblique_binning.split(',')
            obq                   = []
            for i in obq_split:
                a1 = i.replace('{','')
                a2 = a1.replace('}','')
                a3 = int(a2)
                obq.append(a3)
            self.z_pixel_spacing,self.y_pixel_spacing, self.x_pixel_spacing  = self.find_tags(sinogram_header, ['scale factor (mm/pixel) [3]', 'scale factor (degree/pixel) [2]', 'scale factor (mm/pixel) [3]'])
            self.z_pixel_spacing, self.y_pixel_spacing, self.x_pixel_spacing = float(self.z_pixel_spacing), float(self.y_pixel_spacing), float(self.x_pixel_spacing)
            self.oblique_segments = obq # I think this is the binning of oblique sinograms.
            cal_date,             = self.find_tags(sinogram_header, ['%tracer injection date (yyyy:mm:dd)'])
            cal_time,             = self.find_tags(sinogram_header, ['%tracer injection time (hh:mm:ss GMT+00:00)'])
            self.calibration_time = datetime.datetime.strptime(cal_date+cal_time,'%Y:%m:%d%H:%M:%S')
            self.half_life,       = self.find_tags(sinogram_header, ['isotope gamma halflife (sec)'])
            self.activity,        = self.find_tags(sinogram_header, ['tracer activity at time of injection (Bq)'])
            self.duration,        = self.find_tags(sinogram_header, ['%image duration from timing tags (msec)'])
            acquisition_time,     = self.find_tags(sinogram_header, ['%study time (hh:mm:ss GMT+00:00)'])
            acquisition_times.append(acquisition_time)
            x_dim, y_dim, z_dim   = self.find_tags(sinogram_header, ['matrix size[1]', 'matrix size[2]', 'matrix size[3]'])
            x_dim, y_dim, z_dim   = int(x_dim), int(y_dim), int(z_dim)
            bytes_per_pixel,      = self.find_tags(sinogram_header, ['!number of bytes per pixel'])
            data_file_name,       = self.find_tags(sinogram_header, ['!name of data file'])
            nof_tof_bins,         = self.find_tags(sinogram_header, ['%number of TOF time bins'])
            print('Formatting prompt coincidence data into numpy arrays')
            pixel_array           = n.frombuffer(open(self.data_dir + data_file_name, 'rb').read(), dtype = n.int16)
            TOF_bin               = x_dim*y_dim*z_dim
            prompt_images         = []
            for i in range(int(nof_tof_bins)):
                prompt_images.append(pixel_array[TOF_bin*i:TOF_bin*(i+1)].reshape(z_dim,y_dim,x_dim))
            print('Formatting randoms data into numpy array')
            randoms_image = pixel_array[TOF_bin*int(nof_tof_bins):TOF_bin*(int(nof_tof_bins)+1)].reshape([z_dim,y_dim,x_dim])
            #prompt.append(prompt_images)
            
            count_images      = []
            TOF_sum = n.sum(prompt_images,axis=0)
            #Put each oblique sinogram into a separate container
            for i in range(len(self.oblique_segments)):
                ct = []
                start_bound = int(n.sum(self.oblique_segments[:int(i)]))
                end_bound   = start_bound + int(n.sum(self.oblique_segments[int(i)]))
                for j in range(start_bound, end_bound):
                    ct.append(TOF_sum[j])
                count_images.append(ct)
            #pull the first sinogram bin out as the base image.
            base_image = count_images[0]
            #add the oblique sinograms into the base image around the centre pixel.
            for i in range(1,len(count_images)-1):
                mid_index   = len(base_image)//2
                start_index = int(mid_index - (self.oblique_segments[i]//2))
                end_index   = int(mid_index + (self.oblique_segments[i]//2))
                for j in range(start_index, end_index):
                    base_image[j] += count_images[i][j - start_index]
            TOF_Oblique_Binned.append(base_image)


            #Repeat this process for the randoms image.
            random_images = []
            for i in range(len(self.oblique_segments)):
                ct = []
                start_bound = int(n.sum(self.oblique_segments[:int(i)]))
                end_bound   = start_bound + int(n.sum(self.oblique_segments[int(i)]))
                for j in range(start_bound, end_bound):
                    ct.append(randoms_image[j])
                random_images.append(ct)
            #pull the first sinogram bin out as the base image.
            base_image = random_images[0]
            #add the oblique sinograms into the base image around the centre pixel.
            for i in range(1,len(random_images)-1):
                mid_index   = len(base_image)//2
                start_index = int(mid_index - (self.oblique_segments[i]//2))
                end_index   = int(mid_index + (self.oblique_segments[i]//2))
                for j in range(start_index, end_index):
                    base_image[j] += random_images[i][j - start_index]
            random.append(base_image)
    
        self.prompt_images      = prompt
        self.random             = random
        self.acquisition_times  = acquisition_times
        self.TOF_Oblique_Binned = n.array(TOF_Oblique_Binned)
    



    """ This method generates a .pdf file with the reported values as specified in NEMA NU2 2007.
    """
    def generate_report(self, save_directory = '', file_name = 'NEMA_NU2_CountRate_Scatter', system_details = ['Siemens', 'Intevo'], captions = ['',''], delete_tex_files = True):
        #Generate Plots
        fontsize = 22
        fig, (ax0,ax1, ax2) = pyplot.subplots(3,1, figsize=(21,19))
        ax0.scatter(self.activity_conc, self.True_Rate, label = 'Trues')
        ax0.scatter(self.activity_conc, self.Scatter_Rate, label = 'Scatter')
        ax0.scatter(self.activity_conc, self.Random_Rate , label = 'Random')
        ax0.scatter(self.activity_conc, self.Total_Rate,   label = 'Total')
        ax0.scatter(self.activity_conc, self.NEC_Rate  ,   label = 'NECR')
        ax0.plot(self.activity_conc, self.True_Rate)
        ax0.plot(self.activity_conc, self.Scatter_Rate)
        ax0.plot(self.activity_conc, self.Random_Rate)
        ax0.plot(self.activity_conc, self.Total_Rate)
        ax0.plot(self.activity_conc, self.NEC_Rate)
        ax0.legend(fontsize=fontsize)
        ax0.tick_params(axis='both', labelsize=fontsize)
        ax0.grid('on')
        #ax0.set_ylabel('Event Rate ($ms^{-1}$)', fontsize=16)
        #ax0.set_xlabel('Activity concentration ($kBq/ml$)', fontsize=16)
        ax1.scatter(self.activity_conc, self.True_Rate   ,   label = 'Trues'  )
        ax1.scatter(self.activity_conc, self.Scatter_Rate,   label = 'Scatter')
        ax1.scatter(self.activity_conc, self.Random_Rate ,   label = 'Random' )
        ax1.scatter(self.activity_conc, self.Total_Rate  ,   label = 'Total'  )
        ax1.scatter(self.activity_conc, self.NEC_Rate    ,   label = 'NECR'   )
        ax1.plot(self.activity_conc   , self.True_Rate   )
        ax1.plot(self.activity_conc   , self.Scatter_Rate)
        ax1.plot(self.activity_conc   , self.Random_Rate )
        ax1.plot(self.activity_conc   , self.Total_Rate  )
        ax1.plot(self.activity_conc   , self.NEC_Rate    )
        ax1.legend(fontsize=fontsize)
        ax1.tick_params(axis='both', labelsize=fontsize)
        ax1.set_ylim((0, n.amax(self.True_Rate)*1.01))
        ax1.grid('on')
        #ax1.set_ylabel('Event Rate ($ms^{-1}$)', fontsize=16)
        #ax1.set_xlabel('Activity concentration ($kBq/ml$)', fontsize=16)
        ax2.scatter(self.activity_conc, self.Scatter_Fraction, color='b', label = 'Scatter fraction')
        ax2.plot(self.activity_conc, self.Scatter_Fraction, color='b')
        ax2.plot(self.activity_conc, [self.System_Scatter_Fraction for i in range(len(self.activity_conc))], color='r', lw=1.0, label = 'System scatter fraction')
        ax2.set_xlabel('Activity concentration ($kBq/ml$)', fontsize=fontsize)
        ax2.set_ylabel('Acquisition scatter fraction', fontsize=fontsize)
        ax2.tick_params(axis='both', labelsize=fontsize)
        ax2.legend(fontsize=fontsize)
        ax2.grid('on')
        pyplot.tight_layout()
        fig.text(0.01, 0.55, 'Event rate ($ms^{-1}$)', fontsize=fontsize, rotation='vertical')
        #fig.text(0.42, 0.3, 'Activity concentration ($kBq/ml$)', fontsize=fontsize)
        self.count_rate_plot = fig
        
        
        
        #without randoms estimate
        fig, (ax0,ax1, ax2) = pyplot.subplots(3,1, figsize=(21,19))
        ax0.scatter(self.activity_conc, self.True_Rate, label = 'Trues')
        ax0.scatter(self.activity_conc, self.k0_Scatter, label = 'Scatter')
        ax0.scatter(self.activity_conc, self.k0_Random , label = 'Random')
        ax0.scatter(self.activity_conc, self.Total_Rate,   label = 'Total')
        ax0.scatter(self.activity_conc, self.NEC_Rate  ,   label = 'NECR')
        ax0.plot(self.activity_conc, self.True_Rate)
        ax0.plot(self.activity_conc, self.k0_Scatter)
        ax0.plot(self.activity_conc, self.k0_Random)
        ax0.plot(self.activity_conc, self.Total_Rate)
        ax0.plot(self.activity_conc, self.NEC_Rate)
        ax0.legend(fontsize=fontsize)
        ax0.tick_params(axis='both', labelsize=fontsize)
        ax0.grid('on')
        #ax0.set_ylabel('Event Rate ($ms^{-1}$)', fontsize=16)
        #ax0.set_xlabel('Activity concentration ($kBq/ml$)', fontsize=16)
        ax1.scatter(self.activity_conc, self.True_Rate   ,   label = 'Trues'  )
        ax1.scatter(self.activity_conc, self.k0_Scatter,   label = 'Scatter')
        ax1.scatter(self.activity_conc, self.k0_Random ,   label = 'Random' )
        ax1.scatter(self.activity_conc, self.Total_Rate  ,   label = 'Total'  )
        ax1.scatter(self.activity_conc, self.NEC_Rate    ,   label = 'NECR'   )
        ax1.plot(self.activity_conc   , self.True_Rate   )
        ax1.plot(self.activity_conc   , self.k0_Scatter)
        ax1.plot(self.activity_conc   , self.k0_Random )
        ax1.plot(self.activity_conc   , self.Total_Rate  )
        ax1.plot(self.activity_conc   , self.NEC_Rate    )
        ax1.legend(fontsize=fontsize)
        ax1.tick_params(axis='both', labelsize=fontsize)
        ax1.set_ylim((0, n.amax(self.True_Rate)*1.01))
        ax1.grid('on')
        #ax1.set_ylabel('Event Rate ($ms^{-1}$)', fontsize=16)
        #ax1.set_xlabel('Activity concentration ($kBq/ml$)', fontsize=16)
        ax2.scatter(self.activity_conc, self.k0_scatter_fraction, color='b', label = 'Scatter + Random fraction')
        ax2.plot(self.activity_conc, self.k0_scatter_fraction, color='b')
        ax2.plot(self.activity_conc, [self.k0_system_scatter_fraction for i in range(len(self.activity_conc))], color='r', lw=1.0, label = 'System scatter fraction')
        ax2.set_xlabel('Activity concentration ($kBq/ml$)', fontsize=fontsize)
        ax2.set_ylabel('Acquisition scatter fraction', fontsize=fontsize)
        ax2.tick_params(axis='both', labelsize=fontsize)
        ax2.legend(fontsize=fontsize)
        ax2.grid('on')
        fig.text(0.01, 0.65, 'Event rate ($ms^{-1}$)', fontsize=fontsize, rotation='vertical')
        #fig.text(0.42, 0.3, 'Activity concentration ($kBq/ml$)', fontsize=fontsize)
        pyplot.tight_layout()
        self.k0_count_rate_plot = fig
        

        geometry_options = {"tmargin": "1cm", "lmargin": "2cm", "rmargin": "2cm"}
        pylatex_document = Document(geometry_options = geometry_options)
        #Build the header to the document.
        with pylatex_document.create(Tabu("X[l] ")) as first_page_table:
            my_details = MiniPage(width=NoEscape(r"0.6\textwidth"), pos='t!', align='l')
            my_details.append(bold('Auto-generated NEMA NU2 Count rate and Scatter Report'))
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
        with pylatex_document.create(Tabu("X[l] X[c] X[c]")) as table:
            #table.add_caption("Peak true coincidence rate and peak noise equivalent count rate and the corresponding activity concentrations for which they are measured.")
            row1 = ["With Randoms Estimate ",NoEscape(r"maximum value ($ms^{-1}$)"),NoEscape(r"activity concentration ($kBq/ml$)")]
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
        pyplot.figure(self.count_rate_plot.number)
        #Put in the count rate figure.
        with pylatex_document.create(Figure(position = '!h')) as figure:
            figure.add_plot(width = NoEscape(r"\textwidth"))
            figure.add_caption(captions[0])
        #Put in the scatter fraction figure.
        
        pylatex_document.append(NoEscape(r"\clearpage"))
        
        with pylatex_document.create(Figure(position = '!h')) as figure:
            pyplot.figure(self.k0_count_rate_plot.number)
            figure.add_plot(width = NoEscape(r"\textwidth"))
            figure.add_caption(captions[1])
        pylatex_document.generate_pdf(file_name, clean_tex=delete_tex_files)
        self.LaTex_File = pylatex_document



    def __delete_pixels_outside_Ncm(self,image, N, x_pixel_spacing):
        template   = n.zeros(n.shape(image))
        inner_Ncm = (N*10)/x_pixel_spacing #Slices to keep from the centre
        outer_trim = int((len(template[0][0])//2) - inner_Ncm)
        template[:,:,outer_trim:-outer_trim] = image[:,:,outer_trim:-outer_trim]
        return template


    def __delete_pixels_outside_Ncm_2D(self, image, N, x_pixel_spacing):
        template   = n.zeros(n.shape(image))
        inner_Ncm = (N*10)/x_pixel_spacing #Slices to keep from the centre
        outer_trim = int((len(template[0])//2) - inner_Ncm)
        template[:,outer_trim:-outer_trim] = image[:,outer_trim:-outer_trim]
        return template

    def __get_inner_65cm(self, image, z_pixel_spacing):
        z_fov    = z_pixel_spacing*len(image)
        trim_len = z_fov - 120.0 # mm
        print(' FoV length: ', z_fov, ' mm')
        print('Trim length: ', trim_len, ' mm')
        print('z pixel dim: ', z_pixel_spacing, ' mm')
        if trim_len <= 0:
            print('The axial FOV is < 65 cm. We do not need to trim it.')
            return n.array(image)
        else:
            trim_slices = int(trim_len/(2*z_pixel_spacing))
            return n.array(image[trim_slices:-trim_slices])


    def __align_sinogram(self, image, random, x_pixel_spacing):
        bound        = int(120/x_pixel_spacing)
        template     = n.zeros(n.shape(image))
        random_temp  = n.zeros(n.shape(random))
        first_slice  = image[0]
        max_coords   = []
        centre_coord = len(first_slice[0])//2
        for row in first_slice:
            coord = n.argmax(row)
            if coord < bound or coord > n.shape(template)[2] - bound:
                max_coords.append(max_coords[-1])
            else:
                max_coords.append(coord)
    
        for i in range(len(max_coords)):
            template[:, i,centre_coord - bound: centre_coord + bound]  = image[:, i, max_coords[i] - bound: max_coords[i] + bound]
            random_temp[:, i, centre_coord - bound:centre_coord+bound] = random[:,i, max_coords[i] - bound: max_coords[i] + bound]
                #for i in range(len(template)):
                #pyplot.imshow(template[i], vmin = 0, vmax = 100)
                #pyplot.show()
                #pyplot.imshow(image[i], vmin = 0, vmax = 100)
                #pyplot.show()

        return template, random_temp


    def __get_peak_and_scatter(self, image, random, x_pixel_spacing):
        bound                     = 20# 20 mm bound.
        slices_for_bound          = bound/x_pixel_spacing #float. We will linearly interpolate to the 20 mm part.
        centre_px                 = len(image[0])//2
        acquisition_scatters      = []
        acquisition_total         = []
        true_event                = []
        noise_equivalent_estimate = []
        randoms                   = []
        #pyplot.imshow(image, vmin = 0.0, vmax = n.amax(image))
        #pyplot.show()
        for i in range(len(image)):
            row = image[i]
            random_row = random[i]
                    
            interp_x0, interp_x1    = centre_px - slices_for_bound, centre_px + slices_for_bound #Get the pixel value locations of +-20mm
            interp_y0, interp_y1    = n.interp([interp_x0, interp_x1], n.arange(len(row)), row)  #Get the linear interpolated pixel values
            frac_true              = (slices_for_bound - int(slices_for_bound))
            frac_ScR               = (1 - frac_true)
            
            peak_true_scatter_random_counts = n.sum(row[centre_px - int(slices_for_bound): centre_px + int(slices_for_bound)]) + frac_true*((interp_y0 + row[centre_px- int(slices_for_bound)])/2.) + frac_true*((interp_y1 + row[centre_px + int(slices_for_bound)])/2.0) # The total peak counts. Sums everything between the linearly interpolated points

            #Gets the average of the two linear interpolated points (Equivalent of drawing a straight line) and  sums across the total number of peak pixels.
            peak_scatter_random_ave    = ((interp_y0 + interp_y1)/2.0)
            peak_pixels_scatter_random = 2*slices_for_bound
            peak_scatter_random        = peak_scatter_random_ave*peak_pixels_scatter_random
            
            #Sum everything up to the pixel before and after the peak. Add the extra fractional contributions.
            off_peak_scatter_random    = n.sum(row[:centre_px - int(slices_for_bound) - 1]) + n.sum(row[centre_px + int(slices_for_bound) + 1:]) + frac_ScR*((interp_y0 + row[centre_px-int(slices_for_bound) - 1])/2) + frac_ScR*((interp_y1 + row[centre_px + int(slices_for_bound) + 1])/2.)
            
            scatter_randoms            = peak_scatter_random + off_peak_scatter_random
            true_counts                = peak_true_scatter_random_counts - peak_scatter_random
            randoms_estimate           = n.sum(random_row)
            scatter_estimate           = scatter_randoms - randoms_estimate
            total                      = n.sum(row)
            """
            pyplot.plot(n.arange(len(row)), row, color='b')
            pyplot.scatter(n.arange(len(row)), row, color='b')
            pyplot.scatter([centre_px - slices_for_bound, centre_px + slices_for_bound],[CR_l, CR_r], color='r', s=40)
            pyplot.plot([centre_px - slices_for_bound, centre_px + slices_for_bound],[CR_l, CR_r], color='r', lw=3.0)
            pyplot.plot(n.arange(len(random_row)), random_row, color='green')
            pyplot.show()
            """
            #Scatter
            acquisition_scatters.append(scatter_estimate)
            #Total.
            acquisition_total.append(total)
            #Trues.
            true_event.append(true_counts)
            #NEC
            noise_equivalent_estimate.append( (true_counts)*(true_counts/total) )
            #Randoms
            randoms.append(randoms_estimate)
        #Sum over each slice in the image.
        true                  = n.sum(true_event)
        scatter               = n.sum(acquisition_scatters)
        random                = n.sum(randoms)
        NEC                   = n.sum(noise_equivalent_estimate)
        total                 = n.sum(acquisition_total)
        #Scatter fraction
        scatter_fraction      = (scatter/(total - random))
        scatter_randoms       = ((scatter + random)/total)
        return true, scatter, random, NEC, total, scatter_fraction, scatter_randoms

    def analyse_data(self):
        Phantom_Volume       = 22000 # ml
        self.get_sum_sinogram_image()
        concat_prompts       = self.TOF_Oblique_Binned
        True_Rate            = []
        Scatter_Rate         = []
        Random_Rate          = []
        Total_Rate           = []
        NEC_Rate             = []
        TrueC                = []
        ScatterC             = []
        RandomC              = []
        TotalC               = []
        Scatter_Fraction     = []
        Scatter_Randoms      = []
        for i in range(len(concat_prompts)):
            print(self.header_files[i])
            
            #Call a bunch of the "protected" methods to correctly process the data for analysis.
            z_trimmed        = self.__get_inner_65cm(concat_prompts[i], self.pixel_spacings[i][0]) #Set zero pixels outside the 65 cm axial range.
            x_trimmed        = self.__delete_pixels_outside_Ncm(z_trimmed, 12, self.pixel_spacings[i][2]) #Set zero pixels that are +-12 cm outside the centre of the sinogram
            randoms_z_trim   = self.__get_inner_65cm(self.Randoms[i], self.pixel_spacings[i][0]) # Set zero randoms sinogram pixels outside the 65cm axial range
            randoms_trimmed  = self.__delete_pixels_outside_Ncm(randoms_z_trim, 12, self.pixel_spacings[i][2]) #Set zero pixels that are +-12 cm outside the centre of the randoms.
            random           = n.sum(randoms_trimmed) # Sum the total randoms counted in this acquisition.
            aligned_sinogram, aligned_random = self.__align_sinogram(x_trimmed, randoms_trimmed, self.pixel_spacings[i][2]) #Align the phantom sinogram
            projection_image = n.sum(aligned_sinogram, axis = 1) # Project the aligned sinogram along the angle axis.
            projection_rand  = n.sum(aligned_random, axis=1)
            #x_trimmed_40mm   = self.__delete_pixels_outside_Ncm_2D(projection_image, 4, self.pixel_spacings[i][2]) #Delete pixels outside of a +- 20 mm strip containing the aligned sinogram.
            true_counts, scatter_counts, random_counts, NE_counts, total_counts, scatter_fraction, scatter_randoms =  self.__get_peak_and_scatter(projection_image, projection_rand, self.pixel_spacings[i][2]) #Calculate the relevant quantities from the aligned sinogram.
            print(true_counts, scatter_counts, random_counts, NE_counts, total_counts)
            print('Acquisition ' + str(i) + ': ')
            print('        True event rate: ', true_counts/self.acquisition_durations[i])
            print('     Scatter event rate: ', scatter_counts/self.acquisition_durations[i])
            print('      Random event rate: ', random_counts/self.acquisition_durations[i])
            print('       Total event rate: ', total_counts/self.acquisition_durations[i])
            print('  Noise equivalent rate: ', NE_counts/self.acquisition_durations[i])
            print('        Scatter fracton: ', scatter_fraction)
            True_Rate.append(true_counts/self.acquisition_durations[i])
            Scatter_Rate.append(scatter_counts/self.acquisition_durations[i])
            Random_Rate.append(random_counts/self.acquisition_durations[i])
            Total_Rate.append(total_counts/self.acquisition_durations[i])
            NEC_Rate.append(NE_counts/self.acquisition_durations[i])
            Scatter_Fraction.append(scatter_fraction)
            Scatter_Randoms.append(scatter_randoms)
            TrueC.append(true_counts)
            ScatterC.append(scatter_counts)
            RandomC.append(random_counts)
            TotalC.append(total_counts)
        
        #Attach the calculated values to the Scatter object.
        self.True_Rate                      = True_Rate
        self.Scatter_Rate                   = Scatter_Rate
        self.Random_Rate                    = Random_Rate
        self.Total_Rate                     = Total_Rate
        self.NEC_Rate                       = NEC_Rate
        self.activity_conc                  = n.divide(self.activities, 1000*Phantom_Volume)
        self.Scatter_Fraction               = Scatter_Fraction
        self.Scatter_Randoms                = Scatter_Randoms
        self.System_Scatter_Random_Fraction = (n.sum(ScatterC) + n.sum(RandomC))/n.sum(TotalC)
        self.System_Scatter_Fraction        = n.sum(ScatterC[-4:])/(n.sum(TotalC[-4:]) - n.sum(RandomC[-4:]))
        
        #Without randoms estimate.
        
        #Scatter fraction from early scans with randoms rate < 1% of true rate.
        random_to_true_frac = n.divide(Random_Rate, True_Rate)
        print('Random to true fraction: ')
        print('     min: ', n.amin(random_to_true_frac))
        print('     min index: ', n.argmin(random_to_true_frac))
        print('     max: ', n.amax(random_to_true_frac))
        print('     max index: ', n.argmax(random_to_true_frac))

        min_index = n.argmin(random_to_true_frac)-5
        
        k0_scatter_fraction = []
        for i in range(min_index, len(random_to_true_frac)):
            k0_scatter_fraction.append(Scatter_Randoms)
            
        k0_ScatterFraction  = n.amin(k0_scatter_fraction)
        k0_Scatter          = n.multiply((k0_ScatterFraction/(1 - k0_ScatterFraction)),True_Rate)
        k0_Random           = n.subtract(Total_Rate, n.divide(True_Rate, (1 - k0_ScatterFraction)) )
        self.k0_Scatter     = k0_Scatter
        self.k0_Random      = k0_Random
        self.k0_scatter_fraction = n.divide(n.add(k0_Scatter, k0_Random), Total_Rate)
        self.k0_system_scatter_fraction = k0_ScatterFraction
        self.peak_NEC_aconc = self.activity_conc[n.argmax(self.NEC_Rate)]



def main(root_dir):
    #If ran as main, it will analyse a specified root_dir.
    scatter = Scatter(root_dir)
    scatter.analyse_data()
    scatter.generate_report(captions = ['Count rate as a function of activity concentration with system randoms estimate.','Count rate as a function of activity concentration with no system randoms estimate.'], delete_tex_files = False)

import argparse
if __name__  == '__main__':
    usage  = 'cameron.pain@austin.org.au 20200808: Calculate the NEMA NU2 sensitivity.'
    parser = argparse.ArgumentParser(usage = usage)
    parser.add_argument('root_dir', type = str, help = 'Directory containing two directories named "headers" and "data" containing the singogram headers and data files respectively.')
    args = parser.parse_args()
    main(args.root_dir)




