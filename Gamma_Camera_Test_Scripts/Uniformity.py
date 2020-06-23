#!/usr/bin/env python3
#
# Calculates the integral and differential uniformities on the UFOV and CFOV of a flood image. The image post processing and analysis is done according to the NEMA NU1 protocol.
#
# cdp 20190827
#
import time
import numpy as n
import matplotlib.pyplot as pyplot
import matplotlib.dates  as mpl_dates

from scipy.optimize import curve_fit as curve_fit
import pydicom
import os
from scipy.ndimage import zoom as zoom
from scipy.ndimage import convolve as convolve
import sys
#pylatex imports
from pylatex import Document, PageStyle, Head, Foot, MiniPage, \
    StandAloneGraphic, MultiColumn, Tabu, LongTabu, LargeText, HugeText, MediumText, SmallText, LineBreak, \
    NewPage, Tabularx, TextColor, simple_page_number, Package, FlushLeft
from pylatex.utils import bold, NoEscape
from matplotlib2tikz import save as tikzsave


def generate_uniformity_report(uniformity_data, dicom_file, output_path, images, keep_tex_files=True, number_of_images = 2):
    geometry_options = {"head": "0cm", "margin":"0.5in", "bottom":"0.2in", "includeheadfoot": False}
    report           = Document(geometry_options = geometry_options)
    report.append(NoEscape(r"\pagestyle{empty}"))
    #Gather the scan information from the dicom header.
    detectorVec = dicom_file.DetectorVector
    if type(detectorVec) == int:
        detectorVec = [detectorVec]
    else:
        pass

    isotope      = dicom_file.RadiopharmaceuticalInformationSequence[0].RadionuclideCodeSequence[0].CodeMeaning.replace('^','')
    collimator   = dicom_file.DetectorInformationSequence[0].CollimatorGridName
    cameraID     = dicom_file.StationName
    energyLower  = dicom_file.EnergyWindowInformationSequence[0].EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
    energyUpper  = dicom_file.EnergyWindowInformationSequence[0].EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
    meanEnergy   = (energyUpper + energyLower)/2
    energyRange  = str(int((meanEnergy - energyLower)*100/(meanEnergy)))
    imageDim     = str(dicom_file.Rows) + 'x' + str(dicom_file.Columns)
    pixelSize    = str(dicom_file.PixelSpacing[0]) + ',  ' + str(dicom_file.PixelSpacing[1])
    acqDuration  = str(n.round(float(dicom_file.ActualFrameDuration)/1000.0,1))
    acqTime      = str(dicom_file.AcquisitionTime)
    acqDate      = str(dicom_file.AcquisitionDate)
    if len(detectorVec) == 2:
        detector1_counts = n.sum(dicom_file.pixel_array[0])
        detector2_counts = n.sum(dicom_file.pixel_array[1])
        detector1_CR     = n.round(detector1_counts/float(acqDuration),2)
        detector2_CR     = n.round(detector2_counts/float(acqDuration),2)
        countRateVec     = [detector1_CR, detector2_CR]
        countVec         = [detector1_counts, detector2_counts]
    elif len(detectorVec) == 1:
        countVec         = [n.sum(dicom_file.pixel_array)]
        countRateVec     = [countVec[0]/float(acqDuration)]
    else:
        print('\n\n\n')
        print('Error: The dimensions of the input image are not that of a flood image. Check the input data. Expected an image with dimensions of either [slice, row, column] or [row, column]. Instead got ' + str(n.shape(dicom_file.pixel_array)))
        sys.exit()
    if collimator != 'INTRIN':
        int_or_ex = 'Extrinsic'
    else:
        int_or_ex = 'Intrinsic'
    camera_dir    = args.save_root_dir + cameraID
    test_dir      = camera_dir + '/Uniformity/'
    col_dir       = test_dir + int_or_ex
    isotope_dir   = col_dir + '/' + isotope
    #Test for the existance of the data log
    check_dir_existance([args.save_root_dir, camera_dir, test_dir, col_dir, isotope_dir])
    time.sleep(0.1)
    output_path = args.save_root_dir + cameraID + '/Uniformity/' + int_or_ex + '/' + isotope +  '/' + acqDate + '/'
    check_dir_existance([output_path])
    file_name        = 'auto_generated_report'

    report.append(NoEscape(r"\vspace{-0.5cm}"))
    with report.create(MiniPage(width=NoEscape(r"0.67\textwidth"), pos = 'h!')) as report_type:
        with report_type.create(FlushLeft()) as flushed_left:
            flushed_left.append(HugeText(bold("Uniformity Report")))

    with report.create(MiniPage(width=NoEscape(r"0.33\textwidth"), pos = 'l')) as author_information:
        with author_information.create(FlushLeft()) as flushed_left:
            flushed_left.append(SmallText("Auto Generated on: "))
            flushed_left.append(SmallText(NoEscape(r"\today")))
            flushed_left.append(LineBreak())
            flushed_left.append(SmallText("Written by Cameron D Pain"))
            flushed_left.append(LineBreak())
            flushed_left.append(SmallText("cameron.pain@austin.org.au"))
    report.append(LineBreak())
    report.append(NoEscape(r"\hrule"))
    report.append(NoEscape(r"\vspace{1.cm}"))

    for i in range(len(detectorVec)):
        lower_bound = 0.9
        upper_bound = 1.1
        pyplot.imshow(n.divide(images[i], n.average(images[i])), cmap = pyplot.cm.binary, vmin = lower_bound, vmax = upper_bound )
        pyplot.axis('off')
        pyplot.colorbar(cmap = pyplot.cm.binary, orientation='vertical', pad=0.01, fraction = 0.035)
        pyplot.savefig(output_path + 'D' + str(i+1) + '.png')
        pyplot.clf()
        with report.create(MiniPage(width=NoEscape(r"0.33\textwidth"), pos = 't')) as scan_information:
            with scan_information.create(FlushLeft()) as flushed_left:
                flushed_left.append(NoEscape(r'\vspace{-6.4cm}'))
                flushed_left.append(LargeText(bold("Acquisition Information")))
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Camera: "))
                flushed_left.append(cameraID)
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Acquisition date: "))
                flushed_left.append(acqDate)
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Acquisition time "))
                flushed_left.append(acqTime)
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Collimator: "))
                flushed_left.append(str(collimator))
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Image dimensions: "))
                flushed_left.append(imageDim)
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Pixel size: "))
                flushed_left.append(pixelSize + " mm")
                flushed_left.append(NoEscape(r'\vspace{1cm}'))
                flushed_left.append(LineBreak())


                flushed_left.append(LargeText(bold('Detector Information')))
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Detector number: "))
                flushed_left.append(detectorVec[i])
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Count Rate: "))
                flushed_left.append(str(countRateVec[i]) + " cps")
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Acquisition duration: "))
                flushed_left.append(acqDuration + " sec")
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Accumulated counts: "))
                flushed_left.append(countVec[i])
                flushed_left.append(NoEscape(r'\vspace{1cm}'))
                flushed_left.append(LineBreak())




                #Processing information data
                flushed_left.append(LargeText(bold('Processing Information')))
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Smoothing Filter: "))
                flushed_left.append('NEMA NU 1 2007')
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Rebinned pixel size: "))
                flushed_left.append(str(args.rebinned_pixel_size) + ' mm')
                flushed_left.append(LineBreak())
                flushed_left.append(bold("Differential uniformity length: "))
                flushed_left.append("5 px")
                flushed_left.append(LineBreak())

        report.append(NoEscape(r"\hspace{0.5cm}"))
        with report.create(MiniPage(width=NoEscape(r"0.62\textwidth"), pos="!h")) as table_page:
            table_page.append(NoEscape(r"\vspace{-0.3cm}"))
            with table_page.create(Tabu("X[l] X[c] X[c]")) as table:
                row1 = ["Integral Uniformity","",""]
                row2 = ["UFOV", n.round(uniformity_data[0*(i+1)],2), "<5%"]
                row3 = ["CFOV", n.round(uniformity_data[1*(i+1)],2), "<5%"]
                row4 = ["Differential Uniformity", "",""]
                row5 = ["UFOVx", n.round(uniformity_data[2*(i+1)],2), "<2.5%"]
                row6 = ["UFOVy", n.round(uniformity_data[3*(i+1)],2), "<2.5%"]
                row7 = ["CFOVx", n.round(uniformity_data[4*(i+1)],2), "<2.5%"]
                row8 = ["CFOVy", n.round(uniformity_data[5*i+1],2), "<2.5%"]
                table.add_row(row1)
                table.append(NoEscape(r"\hline"))
                table.add_row(row2)
                table.add_row(row3)
                table.append(NoEscape(r"\hline"))
                table.add_row(row4)
                table.add_row(row5)
                table.add_row(row6)
                table.add_row(row7)
                table.add_row(row8)
                table.append(NoEscape(r"\hline"))
            table_page.append(StandAloneGraphic(image_options=NoEscape(r'width=\textwidth'), filename = 'D' + str(i+1) + '.png'))
        if i == 0:
            report.append(NoEscape(r"\vspace{-0.8cm}"))
            report.append(NoEscape(r"\hrule"))
            report.append(NoEscape(r"\vspace{0.8cm}"))
        else:
            pass
        report.generate_pdf(output_path + file_name , clean_tex=keep_tex_files)

def filterProcessNEMA(Image):
    k = n.multiply(n.array([[1,2,1],[2,4,2],[1,2,1]]),1/16.0) # The smoothing kernel specified in NEMA NU 1 2007
    convolveImage = convolve(Image, k,mode='constant',cval= n.average(Image))
    return convolveImage

def integralUniformity(Image):
    max = float(n.amax(Image))
    min = float(n.amin(Image))
    newImageShape = n.shape(Image)
    CFOVmax = float(n.amax(Image[int(newImageShape[0]/8):int((newImageShape[0]*7/8)),int(newImageShape[1]/8):int((newImageShape[1]*7/8))] ))
    CFOVmin = float(n.amin(Image[int(newImageShape[0]/8):int((newImageShape[0]*7/8)),int(newImageShape[1]/8):int((newImageShape[1]*7/8))] ))
    UFOVint =    (  ( (((max-min)/(max+min)) ))  )*100
    CFOVint = (((CFOVmax-CFOVmin)/(CFOVmax+CFOVmin)) )*100
    return UFOVint,CFOVint

def DifferentialUniformity(Image):
    #Image Prep
    #Differential Uniformity Calc UFOV X
    locIntUnif = []
    for i in range(len(Image)):
        for j in range(len(Image[0])-5):
            locIntUnif.append(Image[i,j:j+5])
    diffUnifs  = []
    for localVector in locIntUnif:
        diffUnifs.append( (max(localVector)-min(localVector))/(max(localVector) + min(localVector))   )
    duUFOVx = n.amax(diffUnifs)*100

    #Differential Uniformity Calc CFOV X
    cfovXdim   = int(n.shape(Image)[0]*0.125)
    cfovYdim   = int(n.shape(Image)[1]*0.125)
    cfovImage  = Image[cfovXdim:-cfovXdim, cfovYdim:-cfovYdim]
    locIntUnif = []
    for i in range(len(cfovImage)):
        for j in range(len(cfovImage[0])-5):
            locIntUnif.append(cfovImage[i,j:j+5])
    diffUnifs  = []
    for localVector in locIntUnif:
        diffUnifs.append( (max(localVector)-min(localVector))/(max(localVector) + min(localVector))   )
    duCFOVx = n.amax(diffUnifs)*100

    #Differential Uniformity Calc UFOV Y
    tImage = n.transpose(Image)
    locIntUnif = []
    for i in range(len(tImage)):
        for j in range(len(tImage[0])-5):
            locIntUnif.append(tImage[i,j:j+5])
    diffUnifs  = []
    for localVector in locIntUnif:
        diffUnifs.append( (max(localVector)-min(localVector))/(max(localVector) + min(localVector))   )
    duUFOVy = n.amax(diffUnifs)*100

    #Differential Uniformity Calc CFOV Y
    tcfovXdim   = int(n.shape(tImage)[0]*0.125)
    tcfovYdim   = int(n.shape(tImage)[1]*0.125)
    tcfovImage  = tImage[tcfovXdim:-tcfovXdim, tcfovYdim:-tcfovYdim]
    locIntUnif = []
    for i in range(len(tcfovImage)):
        for j in range(len(tcfovImage[0])-5):
            locIntUnif.append(tcfovImage[i,j:j+5])
    diffUnifs  = []
    for localVector in locIntUnif:
        diffUnifs.append( (max(localVector)-min(localVector))/(max(localVector) + min(localVector))   )
    duCFOVy = n.amax(diffUnifs)*100
    return duUFOVx, duUFOVy, duCFOVx, duCFOVy


def removeZeros(Image):
    centre_pixels = n.divide(n.shape(Image),2).astype(n.int16)
    CFOV          = Image[centre_pixels[0]-int(0.2*centre_pixels[0]):centre_pixels[0] + int(0.2*centre_pixels[0]),  centre_pixels[1]-int(0.2*centre_pixels[1]):centre_pixels[1] + int(0.2*centre_pixels[1])]
    pixel_threshold = 0.75*n.average(CFOV)
    data = []
    for i in range(len(Image)):
        append_row = []
        for j in range(len(Image[0])):
            if Image[i][j] >= pixel_threshold:
                append_row.append(Image[i][j])
            else:
                append_row.append(0)
        data.append(append_row)
    data = n.array(data)
    trimmed_image_1 = []
    for i in range(len(data)):
        if n.amax(data[i]) != 0:
            trimmed_image_1.append(data[i])
    trimmed_image_1 = n.array(trimmed_image_1)
    trimmed_image   = []
    for i in range(len(trimmed_image_1[0])):
        if n.amax(trimmed_image_1[:,i]) != 0:
            trimmed_image.append(trimmed_image_1[:,i])
    trimmed_image = n.array(trimmed_image).transpose()
    return_image = trimmed_image[1:-1,1:-1]
    return return_image



def rebin_image(Image, rebinning_factor):
    range_x, range_y = n.divide(n.shape(Image),rebinning_factor).astype(n.int16)
    data = []
    for i in range(range_y):
        for j in range(range_x):
            data.append( n.sum(Image[i*rebinning_factor:(i+1)*rebinning_factor,j*rebinning_factor:(j+1)*rebinning_factor]) )
    data = n.array(data)
    returnImage = n.reshape(data, [range_x,range_y])
    return returnImage


#Adjust the size of the pixels to 6.4 mm using the ndimage zoom function.
def change_pixel_size(Image, input_pixel_size, output_pixel_size = 6.4):
    rebin_factor   = int(output_pixel_size/input_pixel_size)
    new_pixel_size = rebin_factor * input_pixel_size
    rebinned_image = rebin_image(Image, rebin_factor)
    zoom_factor    = float(output_pixel_size/new_pixel_size)
    zoom_image     = zoom(rebinned_image, 1/zoom_factor)
    return zoom_image


def generate_data_log(Uniformity_Data, dicom_file, keep_tex_files = True):
    print('Updating log file...')
    detectorVec = dicom_file.DetectorVector
    if type(detectorVec) == int:
        detectorVec = [detectorVec]
    else:
        pass
    isotope      = dicom_file.RadiopharmaceuticalInformationSequence[0].RadionuclideCodeSequence[0].CodeMeaning.replace('^','')
    collimator   = dicom_file.DetectorInformationSequence[0].CollimatorGridName
    cameraID     = dicom_file.StationName
    energyLower  = dicom_file.EnergyWindowInformationSequence[0].EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
    energyUpper  = dicom_file.EnergyWindowInformationSequence[0].EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
    acqDuration  = str(n.round(float(dicom_file.ActualFrameDuration)/1000.0,1))
    acqTime      = str(dicom_file.AcquisitionTime)
    acqDate      = str(dicom_file.AcquisitionDate)
    if len(detectorVec) == 2:
        detector1_counts = n.sum(dicom_file.pixel_array[0])
        detector2_counts = n.sum(dicom_file.pixel_array[1])
        detector1_CR     = n.round(detector1_counts/float(acqDuration),2)
        detector2_CR     = n.round(detector2_counts/float(acqDuration),2)
        countRateVec     = [detector1_CR, detector2_CR]
        countVec         = [detector1_counts, detector2_counts]
    elif len(detectorVec) == 1:
        countVec         = [n.sum(dicom_file.pixel_array)]
        countRateVec     = [countVec[0]/float(acqDuration)]
    if collimator != 'INTRIN':
        int_or_ex = 'Extrinsic'
    else:
        int_or_ex = 'Intrinsic'
    print('\n\n\n')
    print('Checking for previous data log entries...')
    camera_dir    = args.save_root_dir + cameraID
    test_dir      = camera_dir + '/Uniformity/'
    col_dir       = test_dir + int_or_ex
    isotope_dir   = col_dir + '/' + isotope
    log_file_path = isotope_dir + '/Data_Log.txt'
    #Test for the existance of the data log
    check_dir_existance([args.save_root_dir, camera_dir, test_dir, col_dir, isotope_dir])
    try:
        log_file = open(log_file_path, 'a+')
        print('log file found.')
    except:
        log_file = open(log_file_path, 'w+')
        print('No previous log file found. Writing a new one.')

    log_file.write(str(acqDate) + ',' + str(acqTime) + '\n')
    for i in range(len(detectorVec)):
        log_file.write('Detector' + str(i + 1) + ',' + str(Uniformity_Data[0*(i+1)]) + ',' + str(Uniformity_Data[1*(i+1)]) + ',' + str(Uniformity_Data[2*(i+1)]) + ',' + str(Uniformity_Data[3*(i+1)]) + ',' + str(Uniformity_Data[4*(i+1)]) + ',' + str(Uniformity_Data[5*(i+1)]) + '\n')
    log_file.write('#\n')
    log_file.close()
    print('log file saved.')
    print('updating plots...')



    log_file  = open(log_file_path, 'r').read().split('#\n')[:-1]

    D1_time_date  = []
    D1_UFOV_int   = []
    D1_CFOV_int   = []
    D1_UFOV_dx    = []
    D1_UFOV_dy    = []
    D1_CFOV_dx    = []
    D1_CFOV_dy    = []

    D2_time_date  = []
    D2_UFOV_int   = []
    D2_CFOV_int   = []
    D2_UFOV_dx    = []
    D2_UFOV_dy    = []
    D2_CFOV_dx    = []
    D2_CFOV_dy    = []

    for entry in log_file:
        entry_split = entry.split('\n')[:-1]
        for i in range(len(entry_split)):
            datum = entry_split[i].split(',')
            d1 = False
            d2 = False
            if datum[0] == 'Detector1':
                d1 = True
                D1_UFOV_int.append(float(datum[1]))
                D1_CFOV_int.append(float(datum[2]))
                D1_UFOV_dx.append(float(datum[3]))
                D1_UFOV_dy.append(float(datum[4]))
                D1_CFOV_dx.append(float(datum[5]))
                D1_CFOV_dy.append(float(datum[6]))
            if datum[0] == 'Detector2':
                d2 = True
                D2_UFOV_int.append(float(datum[1]))
                D2_CFOV_int.append(float(datum[2]))
                D2_UFOV_dx.append(float(datum[3]))
                D2_UFOV_dy.append(float(datum[4]))
                D2_CFOV_dx.append(float(datum[5]))
                D2_CFOV_dy.append(float(datum[6]))
            if d1:
                D1_time_date.append(entry_split[0].split(',')[0] + entry_split[0].split(',')[1][:-3])
            if d2:
                D2_time_date.append(entry_split[0].split(',')[0] + entry_split[0].split(',')[1][:-3])

    D1_tick_indices = [0,int(len(D1_time_date)*0.25), int(len(D1_time_date)*0.5), int(len(D1_time_date)*0.75), int(len(D1_time_date)-1)]
    D2_tick_indices = [0,int(len(D2_time_date)*0.25), int(len(D2_time_date)*0.5), int(len(D2_time_date)*0.75), int(len(D2_time_date)-1)]
    D1_xticks = []
    D2_xticks = []
    for i in range(len(D1_tick_indices)):
        D1_xticks.append(mpl_dates.datestr2num(D1_time_date[D1_tick_indices[i]]))
        D2_xticks.append(mpl_dates.datestr2num(D2_time_date[D2_tick_indices[i]]))


    years  = mpl_dates.YearLocator()
    months = mpl_dates.MonthLocator()
    days   = mpl_dates.DayLocator()



    figure = pyplot.figure(figsize=(8, 4.1))
    ax     = figure.add_subplot(1,1,1)
    ax.xaxis.set_major_locator(days)
    #ax.set_xticks(D1_xticks)
    ax.plot_date(mpl_dates.datestr2num(D1_time_date), D1_UFOV_int, color='b', marker = 'o', label='UFOV')
    ax.plot_date(mpl_dates.datestr2num(D1_time_date), D1_CFOV_int, color='cyan', marker = 's', label= 'CFOV')
    ax.set_ylim((0, 1.8*n.amax([D1_UFOV_int, D2_UFOV_int])))
    ax.set_title('Detector 1 Integral Uniformity')
    ax.set_xlabel('Acquisition date', fontsize=14)
    ax.set_ylabel('Uniformity (%)'  , fontsize=14)
    ax.legend()
    figure.savefig(isotope_dir + '/D1_integral.pdf')
    #tikzsave(isotope_dir + '/UFOV_int.tex', figurewidth='15cm', figureheight='6cm')
    pyplot.clf()
    figure = pyplot.figure(figsize=(8, 4.1))
    ax     = figure.add_subplot(1,1,1)
    ax.xaxis.set_major_locator(days)
    #ax.set_xticks(D1_xticks)
    ax.plot_date(mpl_dates.datestr2num(D1_time_date), D1_UFOV_dx, color='b', marker = 'o', label='UFOV x')
    ax.plot_date(mpl_dates.datestr2num(D1_time_date), D1_UFOV_dy, color='r', marker = 'o', label='UFOV y')
    ax.plot_date(mpl_dates.datestr2num(D1_time_date) ,D1_CFOV_dx, color='cyan', marker = 's', label='CFOV x')
    ax.plot_date(mpl_dates.datestr2num(D1_time_date), D1_CFOV_dy, color='orange', marker = 's', label='CFOV y')
    ax.set_ylim((0, 1.8*n.amax([D1_UFOV_dy, D1_UFOV_dx])))
    ax.set_title('Detector 1 Differential Uniformity')
    ax.set_xlabel('Acquisition date', fontsize=14)
    ax.set_ylabel('Uniformity (%)'  , fontsize=14)
    ax.legend()
    figure.savefig(isotope_dir + '/D1_differential.pdf')
    #tikzsave(isotope_dir + '/D1_UFOV_int.tex', figurewidth='15cm', figureheight='6cm')
    pyplot.clf()
    figure = pyplot.figure(figsize=(8, 4.1))
    ax     = figure.add_subplot(1,1,1)
    ax.xaxis.set_major_locator(days)
    #ax.set_xticks(D2_xticks)
    ax.plot_date(mpl_dates.datestr2num(D2_time_date), D2_UFOV_int, color='b', marker = 'o', label='UFOV')
    ax.plot_date(mpl_dates.datestr2num(D2_time_date), D2_CFOV_int, color='cyan', marker = 's', label='CFOV')
    ax.set_ylim((0, 1.8*n.amax([D1_UFOV_int, D2_UFOV_int])))
    ax.set_title('Detector 2 Integral Uniformity')
    ax.set_xlabel('Acquisition date', fontsize=14)
    ax.set_ylabel('Uniformity (%)'  , fontsize=14)
    ax.legend()
    figure.savefig(isotope_dir + '/D2_integral.pdf')
    #tikzsave(isotope_dir + '/UFOV_int.tex', figurewidth='15cm', figureheight='6cm')
    pyplot.clf()
    figure = pyplot.figure(figsize=(8, 4.1))
    ax     = figure.add_subplot(1,1,1)
    ax.xaxis.set_major_locator(days)
    #ax.set_xticks(D2_xticks)
    ax.plot_date(mpl_dates.datestr2num(D2_time_date), D2_UFOV_dx, color='b'     , marker = 'o', label='UFOV x')
    ax.plot_date(mpl_dates.datestr2num(D2_time_date), D2_UFOV_dy, color='r'     , marker = 'o', label='UFOV y')
    ax.plot_date(mpl_dates.datestr2num(D2_time_date) ,D2_CFOV_dx, color='cyan'  , marker = 's', label='CFOV x')
    ax.plot_date(mpl_dates.datestr2num(D2_time_date), D2_CFOV_dy, color='orange', marker = 's', label='CFOV y')
    ax.set_ylim((0, 1.8*n.amax([D2_UFOV_dx, D2_UFOV_dy])))
    ax.set_title('Detector 2 Differential Uniformity')
    ax.set_xlabel('Acquisition date', fontsize=14)
    ax.set_ylabel('Uniformity (%)'  , fontsize=14)
    ax.legend()
    figure.savefig(isotope_dir + '/D2_differential.pdf')
    #tikzsave(isotope_dir + '/D2_UFOV_int.tex', figurewidth='15cm', figureheight='6cm')
    pyplot.clf()


    geometry_options = {"head": "0cm", "margin":"0.5in", "bottom":"0.2in", "includeheadfoot": False}
    plot_report      = Document(geometry_options = geometry_options)
    plot_report.append(NoEscape(r"\pagestyle{empty}"))
    plot_report.append(NoEscape(r"\vspace{-0.5cm}"))
    with plot_report.create(MiniPage(width=NoEscape(r"0.67\textwidth"), pos = 'h!')) as report_type:
        with report_type.create(FlushLeft()) as flushed_left:
            flushed_left.append(HugeText(bold("Uniformity Trend Analysis")))


    with plot_report.create(MiniPage(width=NoEscape(r"0.33\textwidth"), pos = 'h!')) as author_information:
        with author_information.create(FlushLeft()) as flushed_left:
            flushed_left.append(SmallText("Last updated on: "))
            flushed_left.append(SmallText(NoEscape(r"\today")))
            flushed_left.append(LineBreak())
            flushed_left.append(SmallText("Written by Cameron D Pain"))
            flushed_left.append(LineBreak())
            flushed_left.append(SmallText("cameron.pain@austin.org.au"))
    plot_report.append(LineBreak())
    plot_report.append(NoEscape(r"\hrule"))
    plot_report.append(NoEscape(r"\vspace{1.cm}"))
    print('Generating report')
    trend_files = ['D1_integral.pdf', 'D1_differential.pdf', 'D2_integral.pdf', 'D2_differential.pdf']
    plot_report.append(NoEscape(r'\hspace{-0.5cm}'))
    for image_file in trend_files:
        with plot_report.create(MiniPage(width = NoEscape(r'\textwidth'), pos='l')) as plot:
            plot.append(StandAloneGraphic(image_options=NoEscape(r'width=\textwidth'), filename = image_file))
        plot_report.append(LineBreak())
        plot_report.append(NoEscape(r'\vspace{0.8cm}'))
    plot_report.generate_pdf(isotope_dir + '/trend_analysis_plots' , clean_tex= keep_tex_files)
    time.sleep(0.05)
    os.popen('rm -rf ' + isotope_dir + '/D1_integral.pdf ; rm -rf ' + isotope_dir + '/D1_differential.pdf ; rm -rf ' + isotope_dir + '/D2_integral.pdf ; rm -rf ' + isotope_dir + '/D2_differential.pdf')





def check_dir_existance(directories, build_directories=True):
    directories_exist = []
    for dir in directories:
        test = os.popen('ls ' + dir).read()
        if len(test)!= 0:
            directories_exist.append(True)
        else:
            directories_exist.append(False)
            if build_directories:
                os.popen('mkdir ' + dir)
    time.sleep(0.1)


def main(srcfile, show_images, rebinned_pixel_size, generate_report, save_images, keep_tex_files, update_data_log, save_root_dir):
    check_dir_existance([save_root_dir])
    ds         = pydicom.read_file(srcfile)
    pixelData  = ds.pixel_array

    if len(n.shape(pixelData)) == 3:
        simultaneous_acquisition = True
    elif len(n.shape(pixelData)) == 2:
        simultaneous_acquisition = False
    else:
        print('\n\n\n')
        print('Error: The dimensions of the input data is neither a 2 dimensional image, nor a 3 dimensional data set containing images. Check the input data.')
        sys.exit()

    if simultaneous_acquisition:
        D1         = pixelData[0]
        D2         = pixelData[1]
    else:
        D1         = pixelData

    rebinned_image  = change_pixel_size(D1, float(ds.PixelSpacing[0]), output_pixel_size = rebinned_pixel_size)
    trimmed_image   = removeZeros(rebinned_image)
    filtered_image_D1  = filterProcessNEMA(trimmed_image)
    if show_images:
        pyplot.imshow(D1)
        pyplot.xlabel('Raw data', fontsize=30)
        pyplot.show()
        pyplot.imshow(rebinned_image)
        pyplot.xlabel('Rebinned image', fontsize=30)
        pyplot.show()
        pyplot.imshow(trimmed_image)
        pyplot.xlabel('Edge trimmed image', fontsize=30)
        pyplot.show()
        pyplot.imshow(filtered_image_D1)
        pyplot.xlabel('NEMA NU 1 2007 filtered image.', fontsize=30)
        pyplot.show()
    if save_images != '':
        if save_images[-1] != '/':
            save_images = save_images + '/'
        os.popen('mkdir ' + save_images)
        pyplot.imshow(D1)
        pyplot.xlabel('Raw data', fontsize=20)
        pyplot.savefig( save_images + 'raw_data_image_1.png')
        pyplot.imshow(rebinned_image)
        pyplot.xlabel('Rebinned image', fontsize=20)
        pyplot.savefig( save_images + 'rebinned_image_1.png')
        pyplot.imshow(trimmed_image)
        pyplot.savefig( save_images + 'edge_trimmed_image_1.png')
        pyplot.xlabel('Edge trimmed image', fontsize=20)
        pyplot.imshow(filtered_image_D1)
        pyplot.xlabel('NEMA NU 1 2007 filtered image.', fontsize=20)
        pyplot.savefig( save_images + 'uniformity_calculation_image_1.png')

    IU = integralUniformity(filtered_image_D1)
    DU = DifferentialUniformity(filtered_image_D1)
    print('Detector 1:')
    print(' Integral Uniformity: ')
    print('      UFOV:')
    print('             ' + str(IU[0]) + ' %')
    print('      CFOV:')
    print('             ' + str(IU[1]) + ' %')
    print(' Differential Uniformity: ')
    print('      UFOV (x,y):')
    print('            (' + str(DU[0]) + ',' + str(DU[1]) + ') %')
    print('      CFOV (x,y):')
    print('            (' + str(DU[2]) + ',' + str(DU[3]) + ') %')
    print('')
    D1_Report_Data = [IU[0], IU[1], DU[0], DU[1], DU[2], DU[3]]

    if generate_report:
        if simultaneous_acquisition == False:
            generate_uniformity_report(D1_Report_Data, ds, generate_report, [filtered_image_D1], keep_tex_files = keep_tex_files, number_of_images = 1)
        #    os.popen('rm -rf D1_temp.png')
        else:
            pass
    if update_data_log == True and simultaneous_acquisition == False:
        generate_data_log(D1_Report_Data, ds, keep_tex_files = keep_tex_files)

    if simultaneous_acquisition:
        rebinned_image  = change_pixel_size(D2, float(ds.PixelSpacing[0]), 6.40)
        trimmed_image   = removeZeros(rebinned_image)
        filtered_image_D2  = filterProcessNEMA(trimmed_image)
        if show_images:
            pyplot.imshow(D1)
            pyplot.xlabel('Raw data', fontsize=20)
            pyplot.show()
            pyplot.imshow(rebinned_image)
            pyplot.xlabel('Rebinned image', fontsize=20)
            pyplot.show()
            pyplot.imshow(trimmed_image)
            pyplot.xlabel('Edge trimmed image', fontsize=20)
            pyplot.show()
            pyplot.imshow(filtered_image_D2)
            pyplot.xlabel('NEMA NU 1 2007 filtered image.', fontsize=30)
            pyplot.show()
        if save_images!= '':
            pyplot.imshow(D2)
            pyplot.xlabel('Raw data', fontsize=20)
            pyplot.savefig( save_images + 'raw_data_image_2.png')
            pyplot.imshow(rebinned_image)
            pyplot.xlabel('Rebinned image', fontsize=20)
            pyplot.savefig( save_images + 'rebinned_image_2.png')
            pyplot.imshow(trimmed_image)
            pyplot.savefig( save_images + 'edge_trimmed_image_2.png')
            pyplot.xlabel('Edge trimmed image', fontsize=20)
            pyplot.imshow(filtered_image_D2)
            pyplot.xlabel('NEMA NU 1 2007 filtered image.', fontsize=20)
            pyplot.savefig( save_images + 'uniformity_calculation_image_2.png')

        IU = integralUniformity(filtered_image_D2)
        DU = DifferentialUniformity(filtered_image_D2)
        print('Detector 2:')
        print(' Integral Uniformity: ')
        print('      UFOV:')
        print('             ' + str(IU[0]) + ' %')
        print('      CFOV:')
        print('             ' + str(IU[1]) + ' %')
        print(' Differential Uniformity: ')
        print('      UFOV (x,y):')
        print('            (' + str(DU[0]) + ',' + str(DU[1]) + ') %')
        print('      CFOV (x,y):')
        print('            (' + str(DU[2]) + ',' + str(DU[3]) + ') %')
        print('')
        D2_Report_Data = [IU[0], IU[1], DU[0], DU[1], DU[2], DU[3]]
        Report_Data    = D1_Report_Data + D2_Report_Data
        if generate_report:
            generate_uniformity_report(Report_Data, ds, generate_report, [filtered_image_D1, filtered_image_D2], keep_tex_files = keep_tex_files, number_of_images = 2)
        if update_data_log:
            generate_data_log(Report_Data, ds, keep_tex_files = keep_tex_files)






import argparse

if __name__ == '__main__' :
  usage = 'Written by Cameron Pain. Calculates the integral and differential uniformities on the UFOV and CFOV of a flood image. The image post processing and analysis is done according to the NEMA NU1 protocol.'
  parser = argparse.ArgumentParser(description = usage)
  parser.add_argument('srcfile'              , type = str                                                        , help = 'Specify the path to the .dcm file containing the uniformity images.')
  parser.add_argument('--show_images'        , dest = 'show_images'    , default = False , action = 'store_true' , help = 'Set this flag to show the flood images.')
  parser.add_argument('--rebinned_pixel_size', type = float            , default = 6.4                           , help = 'Set this flag to change the pixel size of the analysed image. The default pixel size is 6.4 as specified in NEMA NU 1 2007.')
  parser.add_argument('--generate_report'    , dest = 'generate_report', default = False , action = 'store_true' , help = 'Specify whether you want to save a report of the output data. It will save to the --save_root_dir')
  parser.add_argument('--save_images'        , type = str              , default = ''                            , help = 'Save images of the processed data. Specifiy a directory name and it will save the appropriate image data into it.')
  parser.add_argument('--keep_tex_files'     , dest = 'keep_tex_files' , default = True  , action = 'store_false', help = 'Set this flag to keep the .tex files from the generated report.' )
  parser.add_argument('--update_data_log'    , dest = 'update_data_log', default = False , action = 'store_true' , help = 'Set this flag to update a running data log. If no log is detected in the --save_root_dir it will create one.')
  parser.add_argument('--save_root_dir'      , type = str              , default = '/Users/cameron/Desktop/QC_Data/'            , help = 'Explicitly specify the directory you wish to save output data to. If not set, data will be saved by default to a directory named QC_Data on the desktop.')
  args = parser.parse_args()
  main(args.srcfile, args.show_images, args.rebinned_pixel_size, args.generate_report, args.save_images, args.keep_tex_files, args.update_data_log, args.save_root_dir)

#end ifls
