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
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, Plot, Figure, Matrix, Alignat, PageStyle, Head, Tabu, MiniPage, Foot, LineBreak
from pylatex.utils import italic, NoEscape, bold



class Sensitivity :
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

    def get_image_data(self):
        self.data_parsed = True
        prompt = []
        random = []
        acquisition_times = []
        for header in self.header_files:
            print('Opening byte data')
            sinogram_header       = open(self.header_dir + header, 'r').read().split('\n')[:-1]
            oblique_binning,      = self.find_tags(sinogram_header , ['%segment table'])
            obq_split             = oblique_binning.split(',')
            obq                   = []
            for i in obq_split:
                a1 = i.replace('{','')
                a2 = a1.replace('}','')
                a3 = float(a2)
                obq.append(a3)
            self.z_pixel_spacing,  = self.find_tags(sinogram_header, ['scale factor (mm/pixel) [3]'])
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
            prompt.append(prompt_images)
            random.append(randoms_image)
        self.prompt_images = prompt
        self.random        = random
        self.acquisition_times = acquisition_times

    def show_sinograms(self, images = 'prompt'):
        if images == 'prompt':
            for i in range(len(self.prompt_images)):
                nof_rows = 4
                nof_cols = 4
                f, ax0   = pyplot.subplots(nof_rows,nof_cols,figsize=(18,10))
                sliderAx = f.add_axes([0.01,0.01,0.8,0.05])
                slider   = Slider(sliderAx, 'Slice', 0, len(self.prompt_images[0][0])-1, valinit = 0, valstep= 1)
                subpts   = []
                f.text(0.45,0.9,'Sleeve configuration ' + str(i))
                for j in range(len(self.prompt_images[0])):
                    subpts.append(ax0[(j//nof_rows)][j%nof_cols].imshow(self.prompt_images[i][j][0], cmap = pyplot.cm.binary, vmin = 0.0, vmax = n.amax(self.prompt_images[i][j])))
                    ax0[(j//nof_rows)][j%nof_cols].set_title('Sinogram ' + str(j))
                def slider_change(val):
                    new_val = int(val)
                    for k in range(len(subpts)):
                        subpts[k].set_data(self.prompt_images[i][k][new_val])
                slider.on_changed(slider_change)
                pyplot.show()
        if images == 'randoms':
            nof_rows = 3
            nof_cols = 3
            f, ax0   = pyplot.subplots(nof_rows,nof_cols,figsize=(18,10))
            sliderAx = f.add_axes([0.01,0.01,0.8,0.05])
            slider   = Slider(sliderAx, 'Slice', 0, len(self.random[0][0])-1, valinit = 0, valstep= 1)
            subpts   = []
            for i in range(len(self.random)):
                subpts.append(ax0[(i//nof_rows)][i%nof_cols].imshow(self.random[i][0], cmap = pyplot.cm.binary, vmin = 0.0, vmax = n.amax(self.random[i])))
                ax0[(i//nof_rows)][i%nof_cols].set_title('Random sinogram ' + str(i))
            def slider_change(val):
                new_val = int(val)
                for k in range(len(subpts)):
                    subpts[k].set_data(self.random[k][new_val])
            slider.on_changed(slider_change)
            pyplot.show()
                
    def calculate_sensitivity(self, show_images = False):
        if self.data_parsed == False:
            print('Execute the method get_image_data() to parse the byte files so the data is available to analyse.')
            return
        #This part uses the sinogram data you parse when you call the get_image_data() method to calculate the sensitivity for each aluminium sleeve configuration and fit the exponential model.
        decay_correction_factor = []
        for i in self.acquisition_times:
            date = datetime.datetime.strptime(i, '%H:%M:%S')
            dt   = (date-self.calibration_time).seconds
            dcf  = 2**(-dt/float(self.half_life))
            decay_correction_factor.append(dcf)
        prompt_sum = []
        for i in range(len(self.prompt_images)):
            prompt_sumi = n.sum(self.prompt_images[i])
            random_sumi = n.sum(self.random[i])
            prompt_sum.append(prompt_sumi - random_sumi)
        sleeve_thickness = [12.5,10,7.5,5.0,2.5]
        activity         = float(self.activity)/1e6   # MBq
        duration         = float(self.duration)/1000.0 # seconds
        print('activity: ', activity, ' MBq')
        print('duration: ', duration, ' seconds')
        print('counts: ', prompt_sum)
        decay_corrected_activity   = n.multiply(activity, decay_correction_factor)
        print('Decay corrected activities: ', decay_corrected_activity)
        decay_corrected_count_rate = n.divide(prompt_sum,n.multiply(activity*duration,decay_correction_factor))
        
        params,cov = curve_fit(sensitivity_fit_function, sleeve_thickness, decay_corrected_count_rate, p0=[5100, 0.1])
        print('fit params: ', params)
        thickness_range = n.arange(0,20, 0.1)
        fit_data        = sensitivity_fit_function(thickness_range, *params)
        f, ax           = pyplot.subplots(1,1,figsize=(18,8))
        ax.plot(thickness_range, fit_data, '--', color='k', lw=3.0)
        ax.scatter(sleeve_thickness, decay_corrected_count_rate, s=80)
        ax.set_xlim((0,15))
        ax.set_ylim((0,1.1*params[0]))
        ax.tick_params(axis='both', labelsize=24)
        ax.set_xlabel('Attenuator Thickness ($mm$)',fontsize=24)
        ax.set_ylabel('Sensitivity ($cps/MBq$)', fontsize=24)
        ax.set_title('Fit Function: $S(x) = S_0 e^{-2\mu_{x}x}$ \nFit Params: $S_0$ = ' + str(n.round(params[0],4)) + ' $cps/MBq$,    $\mu_x$ = ' + str(n.round(params[1],6)) + ' $mm^{-1}$', fontsize=24)
        ax.grid('on')
        self.Al_thickness_sensitivities = decay_corrected_count_rate
        self.system_sensitivity         = params[0]
        self.attenuation_factor         = params[1]
        pyplot.tight_layout()
        self.intrinsic_sensitivity_fit  = f
        if show_images:
            pyplot.show()
        
        
        #This part calculates the sensitivity profile of the scanner. The oblique sinograms are all into the appropriate bins to generate the profile. Note the %segment table tag in the header specifies the starting positions of the oblique sinograms in the data stack.
        
        projection_images = n.sum(self.prompt_images[0],axis=0) #Pull out the smallest sleeve configuration and sum ToF bins
        count_images      = []
        #Put each oblique sinogram into a separate container
        for i in range(len(self.oblique_segments)):
            ct = []
            start_bound = int(n.sum(self.oblique_segments[:int(i)]))
            end_bound   = start_bound + int(n.sum(self.oblique_segments[int(i)]))
            for j in range(start_bound, end_bound):
                ct.append(n.sum(projection_images[j]))
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
        #Convert counts to sensitivity for each slice.
        sensitivity_profile = n.divide(base_image, decay_correction_factor[0]*activity*duration) #divide by duration and decay corrected activity to get counts/second*MBq -> cps/MBq
        f, ax = pyplot.subplots(1,1,figsize=(18,8))
        ax.plot(sensitivity_profile, lw=4.0)
        ax.set_xlabel('Slice index',fontsize=24)
        ax.set_ylabel('Sensitivity ($cps/MBq$)', fontsize=24)
        ax.tick_params(axis='both', labelsize=24)
        ax.set_xlim((0,80))
        ax.grid('on')
        pyplot.tight_layout()
        self.sensitivity_profile = f
        if show_images:
            pyplot.show()
        
    def generate_report(self, save_directory = '', file_name = 'NEMA_NU2_Sensitivity', system_details = ['Siemens', 'Intevo'], captions = ['','']):
        geometry_options = {"tmargin": "1cm", "lmargin": "1cm", "rmargin": "1cm"}
        pylatex_document = Document(geometry_options = geometry_options)
        with pylatex_document.create(Tabu("X[l] ")) as first_page_table:
               my_details = MiniPage(width=NoEscape(r"0.6\textwidth"), pos='t!', align='l')
               my_details.append(bold('Auto-generated NEMA NU2 Sensitivity Report'))
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
        pyplot.figure(self.intrinsic_sensitivity_fit.number)
        with pylatex_document.create(Figure(position = 'h')) as figure:
            figure.add_plot(width = NoEscape(r"0.96\textwidth"))
            figure.append(LineBreak())
            pyplot.figure(self.sensitivity_profile.number)
            figure.add_plot(width = NoEscape(r"0.96\textwidth"))
            figure.add_caption(captions[0])
        pylatex_document.generate_pdf(file_name, clean_tex=True)
        self.LaTex_File = pylatex_document
        
def sensitivity_fit_function(x, R0, mu):
    return R0*n.exp(-mu*2*x)

def main(root_dir, report_name, radial_offset):
    sensitivity = Sensitivity(root_dir)
    sensitivity.get_image_data()
    #sensitivity.show_sinograms(images='randoms')
    sensitivity.calculate_sensitivity(show_images = False)
    sensitivity.generate_report(file_name = report_name, system_details = ['Siemens', radial_offset + ' radial offset','Epworth','Tested: 28/05/2020'], captions = ['Systems sensitivity as a function of attenuator thickness and the axial sensitivity profile. Reported data has been random subtracted.',''])

        
import argparse
if __name__  == '__main__':
    usage  = 'cameron.pain@austin.org.au 20200808: Calculate the NEMA NU2 sensitivity.'
    parser = argparse.ArgumentParser(usage = usage)
    parser.add_argument('root_dir', type = str, help = 'Directory containing the sensitivity data headers.')
    parser.add_argument('radial_offset', type = str, help = 'The radial offset in cm. This is used in the report that is generated.')
    parser.add_argument('report_name', type = str, help = 'Name of the .pdf report you want to generate.')
    args = parser.parse_args()
    main(args.root_dir, args.report_name, args.radial_offset)




